#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dirent.h>

#define INPUT_SIZE   784
#define HIDDEN_SIZE  128
#define OUTPUT_SIZE  10
#define MAX_IMAGES   1000

typedef struct {
    float w1[INPUT_SIZE][HIDDEN_SIZE];
    float b1[HIDDEN_SIZE];
    float w2[HIDDEN_SIZE][OUTPUT_SIZE];
    float b2[OUTPUT_SIZE];
} MLP;

/* ------------------------------------------------------------------ */
/* Activation                                                           */
/* ------------------------------------------------------------------ */
float relu(float x) { return x > 0.0f ? x : 0.0f; }

void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* ------------------------------------------------------------------ */
/* Forward pass                                                         */
/* ------------------------------------------------------------------ */
int forward(MLP *mlp, float *input) {
    float hidden[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        hidden[j] = mlp->b1[j];
        for (int i = 0; i < INPUT_SIZE; i++)
            hidden[j] += input[i] * mlp->w1[i][j];
        hidden[j] = relu(hidden[j]);
    }
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        output[k] = mlp->b2[k];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[k] += hidden[j] * mlp->w2[j][k];
    }
    softmax(output, OUTPUT_SIZE);

    int pred = 0;
    for (int k = 1; k < OUTPUT_SIZE; k++)
        if (output[k] > output[pred]) pred = k;
    return pred;
}

/* ------------------------------------------------------------------ */
/* Chargement poids                                                     */
/* ------------------------------------------------------------------ */
int load_matrix(const char *filename, float *data, int rows, int cols) {
    FILE *f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Erreur : %s\n", filename); return -1; }
    for (int i = 0; i < rows * cols; i++) fscanf(f, "%f", &data[i]);
    fclose(f);
    return 0;
}

int load_vector(const char *filename, float *data, int size) {
    FILE *f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Erreur : %s\n", filename); return -1; }
    for (int i = 0; i < size; i++) fscanf(f, "%f", &data[i]);
    fclose(f);
    return 0;
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
    return 0;
}

/* ------------------------------------------------------------------ */
/* Chargement image BMP 28x28 niveaux de gris                          */
/* ------------------------------------------------------------------ */
int load_bmp_28x28(const char *filename, float *pixels) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Erreur ouverture : %s\n", filename); return -1; }

    /* Vérifier signature BMP */
    unsigned char sig[2];
    fread(sig, 1, 2, f);
    if (sig[0] != 'B' || sig[1] != 'M') {
        fprintf(stderr, "Erreur : %s n'est pas un BMP valide\n", filename);
        fclose(f); return -1;
    }

    /* Offset pixel data */
    fseek(f, 10, SEEK_SET);
    int offset = 0;
    fread(&offset, 4, 1, f);

    /* Largeur et hauteur */
    fseek(f, 18, SEEK_SET);
    int width = 0, height = 0;
    fread(&width,  4, 1, f);
    fread(&height, 4, 1, f);

    if (width != 28 || abs(height) != 28) {
        fprintf(stderr, "Erreur : %s taille %dx%d, attendu 28x28\n", filename, width, abs(height));
        fclose(f); return -1;
    }

    /* Bits par pixel */
    fseek(f, 28, SEEK_SET);
    short bpp = 0;
    fread(&bpp, 2, 1, f);

    /* Padding BMP : chaque ligne alignée sur 4 octets */
    int bytes_per_pixel = bpp / 8;
    int row_size = ((width * bytes_per_pixel + 3) / 4) * 4;
    unsigned char row_buf[256];

    fseek(f, offset, SEEK_SET);

    /* BMP stocké de bas en haut */
    for (int row = 27; row >= 0; row--) {
        fread(row_buf, 1, row_size, f);
        for (int col = 0; col < 28; col++) {
            /* Niveaux de gris : 1 octet si 8bpp, moyenne RGB si 24bpp */
            if (bytes_per_pixel == 1)
                pixels[row * 28 + col] = row_buf[col] / 255.0f;
            else
                pixels[row * 28 + col] = (row_buf[col*3] + row_buf[col*3+1] + row_buf[col*3+2]) / (3.0f * 255.0f);
        }
    }

    fclose(f);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Extraire le label depuis le nom de fichier "5-01.bmp" -> 5          */
/* ------------------------------------------------------------------ */
int get_label(const char *filename) {
    return atoi(filename);  /* atoi s'arrête au premier '-' */
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage : %s <images_dir/> <weights_dir/>\n", argv[0]);
        printf("Exemple : ./MLP_inference custom_digits/ mlp_weights_txt/\n");
        return 1;
    }

    const char *images_dir  = argv[1];
    const char *weights_dir = argv[2];

    /* --- 1. Charger le modèle (hors mesure) --- */
    MLP *mlp = (MLP *)malloc(sizeof(MLP));
    if (!mlp) { fprintf(stderr, "Erreur malloc\n"); return 1; }

    printf("Chargement des poids...\n");
    if (load_mlp(mlp, weights_dir) != 0) { free(mlp); return 1; }

    /* --- 2. Charger toutes les images en mémoire (hors mesure) --- */
    static float images[MAX_IMAGES][INPUT_SIZE];
    int labels[MAX_IMAGES];
    int n_images = 0;

    DIR *dir = opendir(images_dir);
    if (!dir) {
        fprintf(stderr, "Erreur : dossier '%s' introuvable\n", images_dir);
        free(mlp); return 1;
    }

    struct dirent *entry;
    char filepath[512];

    while ((entry = readdir(dir)) != NULL && n_images < MAX_IMAGES) {
        /* Garder uniquement les .bmp */
        char *ext = strrchr(entry->d_name, '.');
        if (!ext || strcasecmp(ext, ".bmp") != 0) continue;

        snprintf(filepath, sizeof(filepath), "%s/%s", images_dir, entry->d_name);

        if (load_bmp_28x28(filepath, images[n_images]) == 0) {
            labels[n_images] = get_label(entry->d_name);
            n_images++;
        }
    }
    closedir(dir);

    if (n_images == 0) {
        fprintf(stderr, "Erreur : aucune image chargée depuis '%s'\n", images_dir);
        free(mlp); return 1;
    }
    printf("%d images chargées\n", n_images);

    /* --- 3. Mesure UNIQUEMENT du forward pass --- */
    int predictions[MAX_IMAGES];
    struct timespec t0, t1;

    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < n_images; i++)
        predictions[i] = forward(mlp, images[i]);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    /* --- 4. Calcul du temps --- */
    long total_ns = (t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec);
    double time_per_image_ms = (double)total_ns / n_images / 1e6;
    double total_ms = (double)total_ns / 1e6;

    /* --- 5. Calcul accuracy --- */
    int correct = 0;
    for (int i = 0; i < n_images; i++)
        if (predictions[i] == labels[i]) correct++;
    float accuracy = (float)correct / n_images * 100.0f;

    /* --- 6. Affichage --- */
    printf("\n========================================\n");
    printf("        BENCHMARK MLP - C\n");
    printf("========================================\n");
    printf("  Nombre d'images         : %d\n", n_images);
    printf("  Temps total (forward)   : %.2f ms\n", total_ms);
    printf("  Temps par image         : %.4f ms\n", time_per_image_ms);
    printf("  Accuracy dataset perso  : %.1f%%\n", accuracy);
    printf("========================================\n");

    free(mlp);
    return 0;
}