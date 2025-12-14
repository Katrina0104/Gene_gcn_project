# 1_image_processor.py
import cv2
import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from PIL import Image
import os

class MedicalImageProcessor:
    def __init__(self):
        print("=== Medical Image Processor ===")
        
    def load_image(self, image_path):
        """Load medical image"""
        print(f"Loading image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ File does not exist: {image_path}")
            return None
        
        # Support multiple image formats
        if image_path.endswith(('.tif', '.tiff')):
            img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print(f"Image dimensions: {img.shape}")
        return img
    
    def preprocess_chromosome_image(self, img):
        """Process chromosome karyotype image"""
        print("Processing chromosome image...")
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Binarization
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find chromosome contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} chromosome contours")
        return enhanced, binary, contours
    
    def extract_chromosome_features(self, img, contours):
        """Extract chromosome features"""
        print("Extracting chromosome features...")
        
        features = []
        chromosome_images = []
        
        for i, contour in enumerate(contours):
            # Calculate contour features
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < 100:  # Small contours may be noise
                continue
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Crop chromosome image
            chromosome_img = img[y:y+h, x:x+w]
            chromosome_images.append(chromosome_img)
            
            # Texture features (simplified version)
            if chromosome_img.size > 0:
                mean_intensity = np.mean(chromosome_img)
                std_intensity = np.std(chromosome_img)
            else:
                mean_intensity = std_intensity = 0
            
            features.append([
                area,
                perimeter,
                aspect_ratio,
                mean_intensity,
                std_intensity,
                w,  # width
                h   # height
            ])
        
        features_array = np.array(features)
        print(f"Extracted features from {len(features)} chromosomes")
        
        return features_array, chromosome_images
    
    def build_chromosome_graph(self, features):
        """Build chromosome relationship graph"""
        print("Building chromosome graph structure...")
        
        num_chromosomes = len(features)
        
        # Create node features (chromosome features)
        x = torch.tensor(features, dtype=torch.float)
        
        # Create edges (based on feature similarity)
        edge_list = []
        
        for i in range(num_chromosomes):
            for j in range(i+1, num_chromosomes):
                # Calculate feature similarity (Euclidean distance)
                similarity = np.exp(-np.linalg.norm(features[i] - features[j]))
                
                if similarity > 0.7:  # Similarity threshold
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # Undirected graph
        
        if len(edge_list) == 0:
            # If insufficient edges, create minimal connected graph
            for i in range(num_chromosomes-1):
                edge_list.append([i, i+1])
                edge_list.append([i+1, i])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        print(f"Graph structure: {num_chromosomes} nodes, {edge_index.shape[1]} edges")
        
        data = Data(x=x, edge_index=edge_index)
        return data
    
    def visualize_results(self, original_img, processed_img, contours, save_path=None):
        """Visualize processing results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(original_img if len(original_img.shape)==3 else 
                         cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Enhanced image
        axes[0, 1].imshow(processed_img, cmap='gray')
        axes[0, 1].set_title('Enhanced Image')
        axes[0, 1].axis('off')
        
        # Contour detection
        contour_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB) if len(original_img.shape)==2 else original_img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        axes[1, 0].imshow(contour_img)
        axes[1, 0].set_title(f'Chromosome Contours ({len(contours)} detected)')
        axes[1, 0].axis('off')
        
        # Feature distribution
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            axes[1, 1].hist(areas, bins=20, alpha=0.7, color='blue')
            axes[1, 1].set_title('Chromosome Area Distribution')
            axes[1, 1].set_xlabel('Area')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization results saved: {save_path}")
        
        plt.show()

def main():
    """Main function: Test image processing"""
    processor = MedicalImageProcessor()
    
    # Test - Create simulated image if no real image exists
    test_image_path = "data/test_chromosome.jpg"
    
    # Check if test image exists
    if os.path.exists(test_image_path):
        img = processor.load_image(test_image_path)
    else:
        print("Creating simulated chromosome image...")
        # Create simulated image
        img = np.zeros((512, 512), dtype=np.uint8)
        # Add simulated chromosomes
        for i in range(46):  # Normal humans have 46 chromosomes
            center = (np.random.randint(100, 412), np.random.randint(100, 412))
            axes = (np.random.randint(10, 30), np.random.randint(5, 15))
            angle = np.random.randint(0, 180)
            cv2.ellipse(img, center, axes, angle, 0, 360, 255, -1)
        
        # Save simulated image
        cv2.imwrite(test_image_path, img)
        img = processor.load_image(test_image_path)
    
    if img is not None:
        # Process image
        enhanced, binary, contours = processor.preprocess_chromosome_image(img)
        
        # Extract features
        features, chromosome_imgs = processor.extract_chromosome_features(enhanced, contours)
        
        # Build graph structure
        if len(features) > 0:
            graph_data = processor.build_chromosome_graph(features)
            print(f"Created graph data: {graph_data}")
            
            # Save graph data
            torch.save(graph_data, 'data/chromosome_graph.pt')
            print("Chromosome graph data saved: data/chromosome_graph.pt")
        
        # Visualize results
        processor.visualize_results(img, enhanced, contours, 'data/image_processing_results.png')

if __name__ == "__main__":
    main()