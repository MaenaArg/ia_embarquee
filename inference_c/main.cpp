/**
 * Pi 5 Camera Stream - MLP Reconnaissance de chiffres
 * Reçoit flux H264, applique inférence MLP, renvoie vers client
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <csignal>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

std::atomic<bool> running(true);
void sigHandler(int) { running = false; }

/* ------------------------------------------------------------------ */
/* MLP                                                                  */
/* ------------------------------------------------------------------ */
#define INPUT_SIZE   784
#define HIDDEN_SIZE  128
#define OUTPUT_SIZE  10
#define BINARIZATION_THR 20

typedef struct {
    float w1[INPUT_SIZE][HIDDEN_SIZE];
    float b1[HIDDEN_SIZE];
    float w2[HIDDEN_SIZE][OUTPUT_SIZE];
    float b2[OUTPUT_SIZE];
} MLP;

float relu_f(float x) { return x > 0.0f ? x : 0.0f; }

void softmax_f(float *x, int n) {
    float mv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mv) mv = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mv); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

int forward(MLP *mlp, float *input, float *probs_out) {
    float hidden[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        hidden[j] = mlp->b1[j];
        for (int i = 0; i < INPUT_SIZE; i++) hidden[j] += input[i] * mlp->w1[i][j];
        hidden[j] = relu_f(hidden[j]);
    }
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        output[k] = mlp->b2[k];
        for (int j = 0; j < HIDDEN_SIZE; j++) output[k] += hidden[j] * mlp->w2[j][k];
    }
    softmax_f(output, OUTPUT_SIZE);
    if (probs_out) for (int k = 0; k < OUTPUT_SIZE; k++) probs_out[k] = output[k];
    int pred = 0;
    for (int k = 1; k < OUTPUT_SIZE; k++) if (output[k] > output[pred]) pred = k;
    return pred;
}

int load_matrix(const char *fn, float *data, int rows, int cols) {
    FILE *f = fopen(fn, "r");
    if (!f) { fprintf(stderr, "Erreur : %s\n", fn); return -1; }
    for (int i = 0; i < rows * cols; i++) { if (fscanf(f, "%f", &data[i]) != 1) break; }
    fclose(f); return 0;
}

int load_vector(const char *fn, float *data, int size) {
    FILE *f = fopen(fn, "r");
    if (!f) { fprintf(stderr, "Erreur : %s\n", fn); return -1; }
    for (int i = 0; i < size; i++) { if (fscanf(f, "%f", &data[i]) != 1) break; }
    fclose(f); return 0;
}

int load_mlp(MLP *mlp, const char *weights_dir) {
    char path[256];
    snprintf(path, sizeof(path), "%s/dense1_weights.txt", weights_dir);
    if (load_matrix(path, (float *)mlp->w1, INPUT_SIZE, HIDDEN_SIZE) != 0) return -1;
    snprintf(path, sizeof(path), "%s/dense1_biases.txt", weights_dir);
    if (load_vector(path, mlp->b1, HIDDEN_SIZE) != 0) return -1;
    snprintf(path, sizeof(path), "%s/dense2_weights.txt", weights_dir);
    if (load_matrix(path, (float *)mlp->w2, HIDDEN_SIZE, OUTPUT_SIZE) != 0) return -1;
    snprintf(path, sizeof(path), "%s/dense2_biases.txt", weights_dir);
    if (load_vector(path, mlp->b2, OUTPUT_SIZE) != 0) return -1;
    printf("Poids MLP charges depuis : %s/\n", weights_dir);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Prétraitement frame -> pixels 28x28 normalisés                      */
/* ------------------------------------------------------------------ */
int preprocess_frame(cv::Mat &frame, float *pixels_out, int thr) {
    cv::Mat gray, inverted, blurred, binary;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::bitwise_not(gray, inverted);
    cv::GaussianBlur(inverted, blurred, cv::Size(3, 3), 0.5);
    cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return -1;

    int max_idx = 0; double max_area = 0;
    for (int i = 0; i < (int)contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) { max_area = area; max_idx = i; }
    }
    if (max_area < 100) return -1;

    cv::Rect bbox = cv::boundingRect(contours[max_idx]);
    int pad = 4;
    bbox.x      = std::max(0, bbox.x - pad);
    bbox.y      = std::max(0, bbox.y - pad);
    bbox.width  = std::min(binary.cols - bbox.x, bbox.width  + 2 * pad);
    bbox.height = std::min(binary.rows - bbox.y, bbox.height + 2 * pad);

    cv::Mat digit = binary(bbox);
    int side = std::max(digit.rows, digit.cols);
    cv::Mat squared = cv::Mat::zeros(side, side, CV_8U);
    digit.copyTo(squared(cv::Rect((side - digit.cols) / 2,
                                  (side - digit.rows) / 2,
                                  digit.cols, digit.rows)));

    cv::Mat resized20, final28;
    cv::resize(squared, resized20, cv::Size(20, 20), 0, 0, cv::INTER_LANCZOS4);
    final28 = cv::Mat::zeros(28, 28, CV_8U);
    resized20.copyTo(final28(cv::Rect(4, 4, 20, 20)));

    for (int r = 0; r < 28; r++)
        for (int c = 0; c < 28; c++)
            pixels_out[r * 28 + c] = final28.at<uchar>(r, c) / 255.0f;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
    signal(SIGINT, sigHandler);
    signal(SIGTERM, sigHandler);

    int inPort  = argc > 1 ? std::stoi(argv[1]) : 5000;
    int outPort = argc > 2 ? std::stoi(argv[2]) : 8554;
    int w       = argc > 3 ? std::stoi(argv[3]) : 1280;
    int h       = argc > 4 ? std::stoi(argv[4]) : 720;
    const char *weights_dir = argc > 5 ? argv[5] : "mlp_weights_txt";

    std::cout << "=== Pi5 Camera + MLP ===" << std::endl;
    std::cout << "In:" << inPort << " Out:" << outPort
              << " " << w << "x" << h
              << " Weights:" << weights_dir << std::endl;

    /* Charger le MLP */
    MLP *mlp = (MLP *)malloc(sizeof(MLP));
    if (!mlp) { fprintf(stderr, "Erreur malloc\n"); return 1; }
    if (load_mlp(mlp, weights_dir) != 0) { free(mlp); return 1; }

    /* Pipelines GStreamer (identiques à l'original) */
    std::string capPipe =
        "tcpclientsrc host=127.0.0.1 port=" + std::to_string(inPort) + " ! "
        "h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 sync=0";

    std::string outPipe =
        "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=15 ! "
        "video/x-h264,profile=baseline ! h264parse config-interval=1 ! "
        "mpegtsmux ! tcpserversink host=0.0.0.0 port=" + std::to_string(outPort);

    cv::VideoCapture cap(capPipe, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) { std::cerr << "Erreur: input" << std::endl; free(mlp); return 1; }

    cv::VideoWriter writer(outPipe, cv::CAP_GSTREAMER, 0, 60, cv::Size(w, h), true);
    if (!writer.isOpened()) { std::cerr << "Erreur: output" << std::endl; free(mlp); return 1; }

    cv::Mat frame;
    int count = 0;
    int prediction = -1;
    float probs[OUTPUT_SIZE] = {0};
    double infer_ms = 0.0;
    double fps = 0.0;
    int thr = BINARIZATION_THR;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    while (running && cap.read(frame)) {
        if (frame.empty()) continue;

        /* === TRAITEMENT OPENCV + MLP === */
        float pixels[INPUT_SIZE];
        auto ti0 = std::chrono::steady_clock::now();
        int ret = preprocess_frame(frame, pixels, thr);
        if (ret == 0)
            prediction = forward(mlp, pixels, probs);
        auto ti1 = std::chrono::steady_clock::now();
        infer_ms = std::chrono::duration<double, std::milli>(ti1 - ti0).count();

        /* Affichage sur la frame */
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(500, 80),
                      cv::Scalar(0, 0, 0), cv::FILLED);

        char txt_pred[64];
        if (prediction >= 0)
            snprintf(txt_pred, sizeof(txt_pred), "Chiffre: %d  (%.0f%%)",
                     prediction, probs[prediction] * 100);
        else
            snprintf(txt_pred, sizeof(txt_pred), "Aucun chiffre detecte");

        cv::putText(frame, txt_pred, cv::Point(10, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    (prediction >= 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 100, 255), 2);

        char txt_info[64];
        snprintf(txt_info, sizeof(txt_info), "FPS: %.0f | Inference: %.2f ms | Seuil: %d", fps, infer_ms, thr);
        cv::putText(frame, txt_info, cv::Point(10, 65),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(200, 200, 200), 1);

        writer.write(frame);
        count++;

        /* Log FPS toutes les secondes */
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> dt = now - t0;
        if (dt.count() >= 1.0) {
            fps = count / dt.count();
            std::cout << "FPS: " << static_cast<int>(fps)
                      << " | Inference: " << infer_ms << " ms"
                      << " | Prediction: " << prediction << std::endl;
            count = 0;
            t0 = now;
        }
    }

    free(mlp);
    return 0;
}