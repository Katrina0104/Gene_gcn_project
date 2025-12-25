import cv2
import numpy as np
import os

class ChromosomeImageProcessor:
    def __init__(self, min_area=500, min_aspect_ratio=2.5):
        self.min_area = min_area
        self.min_aspect_ratio = min_aspect_ratio

    def process(self, image_path, debug_dir="debug"):
        os.makedirs(debug_dir, exist_ok=True)

        # 1. Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("❌ Cannot read image")

        vis = img.copy()

        # 2. Gray + blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Adaptive threshold (染色體 → 白)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5
        )

        # 4. Morphology (接起細長物)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 5. Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        chromosomes = []
        features = []

        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = h / (w + 1e-6)

            # 6. 幾何過濾（關鍵）
            if area < self.min_area:
                continue
            if aspect_ratio < self.min_aspect_ratio:
                continue

            chromosomes.append(c)

            # draw box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # simple features (for later GCN)
            features.append({
                "area": area,
                "width": w,
                "height": h,
                "aspect_ratio": aspect_ratio
            })

        # 7. Save debug images
        cv2.imwrite(os.path.join(debug_dir, "1_gray.png"), gray)
        cv2.imwrite(os.path.join(debug_dir, "2_thresh.png"), thresh)
        cv2.imwrite(os.path.join(debug_dir, "3_detected.png"), vis)

        print(f"✅ Detected chromosomes: {len(chromosomes)}")

        return {
            "count": len(chromosomes),
            "features": features,
            "visualization": vis
        }


if __name__ == "__main__":
    processor = ChromosomeImageProcessor()
    result = processor.process("test_chromosome.jpg")

    print("Chromosome count:", result["count"])
    for i, f in enumerate(result["features"]):
        print(f"#{i+1}", f)
