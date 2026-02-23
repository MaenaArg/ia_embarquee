from PIL import Image, ImageFilter
import os
import numpy as np
from scipy import ndimage

def convert_minimal_bw_improved(input_path, output_path, thr=20):
    # 1. Charger et inverser
    img = Image.open(input_path).convert("L")
    img = Image.eval(img, lambda p: 255 - p)
    
    # 2. Redimensionner en 56x56 d'abord (pour mieux traiter)
    img = img.resize((56, 56), Image.LANCZOS)
    
    # 3. Léger flou pour lisser
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # 4. Binarisation
    img = img.point(lambda p: 255 if p >= thr else 0)
    
    # 5. Convertir en array pour épaissir
    arr = np.array(img, dtype=np.uint8)
    
    # 6. Épaississement LÉGER des traits (dilatation minimale)
    if arr.max() > 0:  # Seulement si l'image n'est pas vide
        from scipy.ndimage import binary_dilation
        arr_bin = arr > 0
        arr_bin = binary_dilation(arr_bin, iterations=2)  # Seulement 2 itérations
        arr = arr_bin.astype(np.uint8) * 255
    
    # 7. Léger lissage
    arr = ndimage.gaussian_filter(arr.astype(np.float32), sigma=0.8)
    
    # 8. Redimensionner à 28x28
    img = Image.fromarray(arr.astype(np.uint8))
    img = img.resize((28, 28), Image.LANCZOS)
    
    # 9. Centrage simple (comme MNIST)
    arr = np.array(img, dtype=np.float32)
    
    # Trouver le contenu
    threshold = 20
    rows = np.any(arr > threshold, axis=1)
    cols = np.any(arr > threshold, axis=0)
    
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Extraire avec padding
        pad = 1
        rmin = max(0, rmin - pad)
        rmax = min(27, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(27, cmax + pad)
        
        digit = arr[rmin:rmax+1, cmin:cmax+1]
        h, w = digit.shape
        
        # Redimensionner si trop grand
        if max(h, w) > 20:
            scale = 20.0 / max(h, w)
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            digit_img = Image.fromarray(digit.astype(np.uint8))
            digit_img = digit_img.resize((new_w, new_h), Image.LANCZOS)
            digit = np.array(digit_img, dtype=np.float32)
            h, w = digit.shape
        
        # Centrer
        centered = np.zeros((28, 28), dtype=np.float32)
        y_offset = (28 - h) // 2
        x_offset = (28 - w) // 2
        centered[y_offset:y_offset+h, x_offset:x_offset+w] = digit
        arr = centered
    
    # 10. Normaliser et sauvegarder
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img_final = Image.fromarray(arr)
    img_final.save(output_path)

def batch_minimal(input_dir, output_dir, thr=20):
    """
    Traite toutes les images avec rapport détaillé
    """
    total_images = 0
    images_traitees = 0
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("CONVERSION AU FORMAT MNIST")
    print("="*60)
    
    for f in sorted(os.listdir(input_dir)):
        if f.lower().endswith(".bmp"):
            total_images += 1
            inp = os.path.join(input_dir, f)
            out = os.path.join(output_dir, f"{os.path.splitext(f)[0]}.bmp")
            
            try:
                convert_minimal_bw_improved(inp, out, thr=thr)
                images_traitees += 1
                
                if images_traitees % 50 == 0:
                    print(f"Progression: {images_traitees}/{total_images}")
            except Exception as e:
                print(f"Erreur avec {f}: {e}")
    
    print(f"\n✓ Images traitées : {images_traitees}/{total_images}")
    print(f"✓ Dossier de sortie : {output_dir}")
    print("="*60)

def visualize_comparison(input_dir, output_dir, n_samples=10):
    """
    Compare avant/après
    """
    import matplotlib.pyplot as plt
    
    files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(".bmp")]
    
    # Prendre des exemples variés
    step = max(1, len(files) // n_samples)
    files = files[::step][:n_samples]
    
    fig, axes = plt.subplots(2, n_samples, figsize=(16, 5))
    
    for i, f in enumerate(files):
        # Original
        img_orig = Image.open(os.path.join(input_dir, f)).convert("L")
        axes[0, i].imshow(img_orig, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12, fontweight='bold')
        
        # Traité
        img_proc = Image.open(os.path.join(output_dir, f"{os.path.splitext(f)[0]}.bmp")).convert("L")
        axes[1, i].imshow(img_proc, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('MNIST-style', fontsize=12, fontweight='bold')
        
        # Label
        label = f.split('-')[0]
        axes[0, i].set_title(f'{label}', fontsize=11, fontweight='bold')
    
    plt.suptitle('Comparaison: Original vs MNIST-style', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparaison_preprocessing.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparaison sauvegardée: comparaison_preprocessing.png")

def check_empty_images(output_dir):
    empty_count = 0
    empty_files = []
    
    for f in sorted(os.listdir(output_dir)):
        if f.lower().endswith(".bmp"):
            img = np.array(Image.open(os.path.join(output_dir, f)).convert("L"))
            if np.count_nonzero(img > 20) < 30:
                empty_count += 1
                empty_files.append(f)
    
    if empty_count > 0:
        print(f"\n⚠️  {empty_count} images avec très peu de pixels:")
        for f in empty_files[:10]:
            print(f"    - {f}")
        if len(empty_files) > 10:
            print(f"    ... et {len(empty_files)-10} autres")
    else:
        print(f"\n✓ Aucune image vide détectée !")
    
    return empty_count

if __name__ == "__main__":
    INPUT_DIR = "data_perso"
    OUTPUT_DIR = "custom_digits"
    
    # Traiter
    batch_minimal(INPUT_DIR, OUTPUT_DIR, thr=20)
    
    # Visualiser
    print("\nCréation de la comparaison visuelle...")
    visualize_comparison(INPUT_DIR, OUTPUT_DIR, n_samples=10)
    
    # Vérifier les images vides
    check_empty_images(OUTPUT_DIR)
    
    print(f"\n✓ TERMINÉ ! Utilisez maintenant le dataset dans le dossier '{OUTPUT_DIR}'.")