import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from datetime import datetime

# Move existing class definitions here to avoid import issues
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
            if img is not None and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img is None:
            print(f"❌ Unable to load image: {image_path}")
            return None
        
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
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Crop chromosome image
            if y+h <= img.shape[0] and x+w <= img.shape[1]:
                chromosome_img = img[y:y+h, x:x+w]
                chromosome_images.append(chromosome_img)
            else:
                chromosome_img = np.array([])
            
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
        
        if features:
            features_array = np.array(features)
        else:
            features_array = np.array([]).reshape(0, 7)
        
        print(f"Extracted features from {len(features)} chromosomes")
        
        return features_array, chromosome_images
    
    def build_chromosome_graph(self, features):
        """Build chromosome relationship graph"""
        print("Building chromosome graph structure...")
        
        num_chromosomes = len(features)
        
        if num_chromosomes == 0:
            print("❌ No chromosome features available for graph construction")
            return None
        
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
        
        if len(edge_list) == 0:
            # If only one chromosome, create self-loop
            edge_list.append([0, 0])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        print(f"Graph structure: {num_chromosomes} nodes, {edge_index.shape[1]} edges")
        
        data = Data(x=x, edge_index=edge_index)
        return data
    
    def visualize_results(self, original_img, processed_img, contours, save_path=None):
        """Visualize processing results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        if len(original_img.shape) == 3:
            axes[0, 0].imshow(original_img)
        else:
            axes[0, 0].imshow(original_img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Enhanced image
        axes[0, 1].imshow(processed_img, cmap='gray')
        axes[0, 1].set_title('Enhanced Image')
        axes[0, 1].axis('off')
        
        # Contour detection
        if len(original_img.shape) == 2:
            contour_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        else:
            contour_img = original_img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        axes[1, 0].imshow(contour_img)
        axes[1, 0].set_title(f'Chromosome Contours ({len(contours)} detected)')
        axes[1, 0].axis('off')
        
        # Feature distribution
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 0]
            if areas:
                axes[1, 1].hist(areas, bins=min(20, len(areas)), alpha=0.7, color='blue')
                axes[1, 1].set_title('Chromosome Area Distribution')
                axes[1, 1].set_xlabel('Area')
                axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization results saved: {save_path}")
        
        plt.show()

class MedicalImageGCN(nn.Module):
    def __init__(self, num_features, num_classes=5):
        """Medical Image GCN Model
        
        Parameters:
            num_features: Input feature dimension
            num_classes: Number of output classes (Normal, Chromosomal Abnormality, Gene Deletion, Structural Variation, Other Abnormality)
        """
        super(MedicalImageGCN, self).__init__()
        
        # GCN layers for analyzing chromosome relationships
        self.gcn1 = GCNConv(num_features, 128)
        self.gcn2 = GCNConv(128, 256)
        self.gcn3 = GCNConv(256, 128)
        
        # CNN feature extractor (for image features)
        self.cnn_backbone = models.resnet18(pretrained=True)
        self.cnn_backbone.fc = nn.Identity()  # Remove final classification layer
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + 512, 256),  # GCN features + CNN features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Anomaly detection head
        self.anomaly_detector = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output anomaly probability
        )
    
    def forward(self, x, edge_index, batch=None, image_features=None):
        # GCN processing of graph structure
        gcn_features = self.gcn1(x, edge_index)
        gcn_features = F.relu(gcn_features)
        gcn_features = F.dropout(gcn_features, training=self.training)
        
        gcn_features = self.gcn2(gcn_features, edge_index)
        gcn_features = F.relu(gcn_features)
        gcn_features = F.dropout(gcn_features, training=self.training)
        
        gcn_features = self.gcn3(gcn_features, edge_index)
        
        # Global pooling
        if batch is not None:
            gcn_features = global_mean_pool(gcn_features, batch)
        else:
            # If no batch information, use mean pooling
            gcn_features = torch.mean(gcn_features, dim=0, keepdim=True)
        
        # If image features exist, perform fusion
        if image_features is not None:
            # Extract CNN features
            cnn_features = self.cnn_backbone(image_features)
            
            # Feature fusion
            combined = torch.cat([gcn_features, cnn_features], dim=1)
            fused_features = self.fusion(combined)
        else:
            fused_features = gcn_features
        
        # Classification prediction
        class_predictions = self.classifier(fused_features)
        
        # Anomaly detection
        anomaly_scores = self.anomaly_detector(fused_features)
        
        return class_predictions, anomaly_scores

class MedicalReportGenerator:
    def __init__(self):
        self.disease_database = {
            'down_syndrome': {
                'name': 'Down Syndrome',
                'chromosome': 'Chromosome 21',
                'abnormality': 'Trisomy',
                'symptoms': ['Intellectual disability', 'Characteristic facial features', 'Heart defects', 'Low muscle tone'],
                'risk_factors': ['Advanced maternal age', 'Family history'],
                'tests': ['Amniocentesis', 'Chorionic villus sampling', 'NIPT']
            },
            'turner_syndrome': {
                'name': 'Turner Syndrome',
                'chromosome': 'X Chromosome',
                'abnormality': 'Monosomy',
                'symptoms': ['Short stature', 'Ovarian dysgenesis', 'Webbed neck', 'Heart abnormalities'],
                'risk_factors': ['Random occurrence'],
                'tests': ['Chromosomal karyotyping']
            },
            'klinefelter': {
                'name': 'Klinefelter Syndrome',
                'chromosome': 'Sex Chromosomes',
                'abnormality': 'XXY',
                'symptoms': ['Small testes', 'Infertility', 'Gynecomastia', 'Learning difficulties'],
                'risk_factors': ['Random occurrence'],
                'tests': ['Chromosomal karyotyping']
            },
            'cri_du_chat': {
                'name': 'Cri du Chat Syndrome',
                'chromosome': 'Chromosome 5',
                'abnormality': 'Deletion',
                'symptoms': ['Cat-like cry', 'Microcephaly', 'Intellectual disability', 'Growth retardation'],
                'risk_factors': ['Random occurrence'],
                'tests': ['Chromosomal microarray analysis']
            }
        }
    
    def generate_report(self, predictions, anomaly_scores, image_info):
        """Generate medical report"""
        
        report = {
            'patient_info': {
                'sample_id': image_info.get('sample_id', 'Unknown'),
                'image_type': image_info.get('image_type', 'Chromosome Karyotype'),
                'analysis_date': image_info.get('date', datetime.now().strftime('%Y-%m-%d'))
            },
            'analysis_results': {
                'abnormality_detected': anomaly_scores.item() if torch.is_tensor(anomaly_scores) else anomaly_scores > 0.5,
                'anomaly_score': float(anomaly_scores.item() if torch.is_tensor(anomaly_scores) else anomaly_scores),
                'predicted_class': torch.argmax(predictions).item() if torch.is_tensor(predictions) else int(np.argmax(predictions)),
                'class_probabilities': F.softmax(predictions, dim=1).tolist() if torch.is_tensor(predictions) else predictions.tolist()
            },
            'detected_abnormalities': [],
            'recommendations': [],
            'risk_assessment': 'Low Risk'
        }
        
        # Add detected abnormalities based on prediction results
        anomaly_score = report['analysis_results']['anomaly_score']
        if anomaly_score > 0.7:
            predicted_class = report['analysis_results']['predicted_class']
            
            if predicted_class == 1:  # Chromosomal numerical abnormality
                report['detected_abnormalities'].append({
                    'type': 'Chromosomal Numerical Abnormality',
                    'description': 'Detected abnormal chromosome count, possibly aneuploidy',
                    'possible_conditions': ['Down Syndrome', 'Edwards Syndrome', 'Patau Syndrome']
                })
                report['recommendations'].append('Perform chromosomal karyotyping for confirmation')
                report['risk_assessment'] = 'Moderate to High Risk'
            
            elif predicted_class == 2:  # Gene deletion/duplication
                report['detected_abnormalities'].append({
                    'type': 'Chromosomal Structural Abnormality',
                    'description': 'Detected chromosomal segment deletion or duplication',
                    'possible_conditions': ['Cri du Chat Syndrome', 'Williams Syndrome']
                })
                report['recommendations'].append('Perform chromosomal microarray analysis (CMA)')
                report['risk_assessment'] = 'Moderate Risk'
        
        # Add standard recommendations
        report['recommendations'].extend([
            'Consult a genetic counselor',
            'Perform further genetic testing for confirmation',
            'Regular follow-up examinations'
        ])
        
        return report
    
    def save_report_as_pdf(self, report, filename='medical_report.pdf'):
        """Save report as PDF"""
        try:
            from fpdf import FPDF
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Title
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="Genetic Image Analysis Report", ln=1, align='C')
            pdf.ln(10)
            
            # Patient Information
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Patient Information", ln=1)
            pdf.set_font("Arial", size=12)
            for key, value in report['patient_info'].items():
                pdf.cell(200, 10, txt=f"{key}: {value}", ln=1)
            pdf.ln(5)
            
            # Analysis Results
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Analysis Results", ln=1)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Abnormality Detected: {'Yes' if report['analysis_results']['abnormality_detected'] else 'No'}", ln=1)
            pdf.cell(200, 10, txt=f"Anomaly Score: {report['analysis_results']['anomaly_score']:.3f}", ln=1)
            pdf.ln(5)
            
            # Detected Abnormalities
            if report['detected_abnormalities']:
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Detected Abnormalities", ln=1)
                pdf.set_font("Arial", size=12)
                for abn in report['detected_abnormalities']:
                    pdf.cell(200, 10, txt=f"Type: {abn['type']}", ln=1)
                    pdf.multi_cell(0, 10, txt=f"Description: {abn['description']}")
                    pdf.ln(5)
            
            # Recommendations
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Recommendations", ln=1)
            pdf.set_font("Arial", size=12)
            for i, rec in enumerate(report['recommendations'], 1):
                pdf.cell(200, 10, txt=f"{i}. {rec}", ln=1)
            
            # Risk Assessment
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Risk Assessment", ln=1)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Overall Risk: {report['risk_assessment']}", ln=1)
            
            # Footer
            pdf.ln(20)
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(200, 10, txt="This report is AI-assisted and for reference only. Please consult a professional physician for diagnosis.", ln=1, align='C')
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            pdf.output(filename)
            print(f"✅ Report saved as: {filename}")
            
        except ImportError:
            print("❌ Requires fpdf installation: pip install fpdf")
            # Save as text file
            txt_filename = filename.replace('.pdf', '.txt')
            os.makedirs(os.path.dirname(txt_filename), exist_ok=True)
            with open(txt_filename, 'w', encoding='utf-8') as f:
                import json
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"✅ Report saved as text file: {txt_filename}")

class IntegratedMedicalSystem:
    def __init__(self, model_path=None):
        """Integrated Medical Image Analysis System"""
        print("=== Genetic Image Analysis System ===")
        print("Version: 1.0")
        print("Features: Chromosome Analysis, Anomaly Detection, Disease Prediction, Report Generation")
        print("-" * 40)
        
        # Initialize components
        self.image_processor = MedicalImageProcessor()
        self.gcn_model = MedicalImageGCN(num_features=7)
        self.report_generator = MedicalReportGenerator()
        
        # Load pre-trained model (if available)
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Disease database
        self.disease_info = {
            0: {'name': 'Normal', 'risk': 'Low', 'description': 'No significant chromosomal abnormalities'},
            1: {'name': 'Chromosomal Numerical Abnormality', 'risk': 'High', 
                'diseases': ['Down Syndrome', 'Edwards Syndrome', 'Patau Syndrome'],
                'description': 'Detected chromosomal aneuploidy'},
            2: {'name': 'Chromosomal Structural Abnormality', 'risk': 'Moderate',
                'diseases': ['Cri du Chat Syndrome', 'Williams Syndrome', 'DiGeorge Syndrome'],
                'description': 'Detected chromosomal deletions, duplications, or rearrangements'},
            3: {'name': 'Sex Chromosome Abnormality', 'risk': 'Low to Moderate',
                'diseases': ['Turner Syndrome', 'Klinefelter Syndrome', 'Triple X Syndrome'],
                'description': 'Sex chromosome numerical or structural abnormalities'},
            4: {'name': 'Other Genetic Abnormality', 'risk': 'Variable',
                'diseases': ['Microdeletion Syndromes', 'Uniparental Disomy'],
                'description': 'Other types of genetic abnormalities'}
        }
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.gcn_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")
            print("⚠️  Using randomly initialized model")
    
    def process_image_file(self, image_path, patient_info=None):
        """Process single image file"""
        print(f"\n📷 Starting image analysis: {image_path}")
        
        # Load image
        img = self.image_processor.load_image(image_path)
        if img is None:
            return None
        
        # Process image
        enhanced, binary, contours = self.image_processor.preprocess_chromosome_image(img)
        
        # Extract features
        features, chromosome_imgs = self.image_processor.extract_chromosome_features(enhanced, contours)
        
        if len(features) == 0:
            print("❌ No chromosomes detected, please check image quality")
            return None
        
        # Build graph structure
        graph_data = self.image_processor.build_chromosome_graph(features)
        
        if graph_data is None:
            print("❌ Unable to build graph structure")
            return None
        
        # Analyze
        result = self.analyze_chromosomes(graph_data, features, patient_info)
        
        # Visualize
        results_dir = 'data/results'
        os.makedirs(results_dir, exist_ok=True)
        self.image_processor.visualize_results(
            img, enhanced, contours, 
            save_path=os.path.join(results_dir, f"{os.path.basename(image_path)}_analysis.png")
        )
        
        return result
    
    def analyze_chromosomes(self, graph_data, features, patient_info=None):
        """Analyze chromosomes"""
        print("\n🔬 Starting chromosome analysis...")
        
        # Prepare model input
        x = graph_data.x
        edge_index = graph_data.edge_index
        
        # Set model to evaluation mode
        self.gcn_model.eval()
        
        with torch.no_grad():
            # Make predictions
            class_predictions, anomaly_scores = self.gcn_model(x, edge_index)
            
            # Get prediction results
            predicted_class = torch.argmax(class_predictions, dim=1).item()
            anomaly_score = anomaly_scores.mean().item()
            
            print(f"📊 Analysis Results:")
            print(f"   - Chromosomes detected: {len(features)}")
            print(f"   - Predicted class: {self.disease_info[predicted_class]['name']}")
            print(f"   - Anomaly score: {anomaly_score:.3f}")
            print(f"   - Risk level: {self.disease_info[predicted_class]['risk']}")
            
            # Calculate statistics
            stats = self.calculate_statistics(features)
            
            # Generate report
            report = self.generate_comprehensive_report(
                predicted_class, anomaly_score, stats, patient_info
            )
            
            return {
                'predictions': class_predictions.numpy(),
                'anomaly_score': anomaly_score,
                'predicted_class': predicted_class,
                'statistics': stats,
                'report': report
            }
    
    def calculate_statistics(self, features):
        """Calculate statistical information"""
        stats = {
            'total_chromosomes': len(features),
            'average_area': float(np.mean(features[:, 0])),
            'average_length': float(np.mean(features[:, 5])),  # Width as length estimate
            'size_variation': float(np.std(features[:, 0]) / np.mean(features[:, 0])),
            'aspect_ratio_stats': {
                'mean': float(np.mean(features[:, 2])),
                'std': float(np.std(features[:, 2])),
                'min': float(np.min(features[:, 2])),
                'max': float(np.max(features[:, 2]))
            }
        }
        
        # Detect potential abnormalities
        if len(features) != 46:
            stats['chromosome_count_abnormal'] = True
            stats['expected_count'] = 46
            stats['difference'] = abs(46 - len(features))
        else:
            stats['chromosome_count_abnormal'] = False
        
        return stats
    
    def generate_comprehensive_report(self, predicted_class, anomaly_score, stats, patient_info):
        """Generate detailed report"""
        
        # Basic report
        report = {
            'report_id': f"REP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'patient_info': patient_info or {
                'patient_id': 'Not provided',
                'age': 'Not provided',
                'gender': 'Not provided',
                'sample_type': 'Chromosome Karyotype'
            },
            'analysis_summary': {
                'result': self.disease_info[predicted_class]['name'],
                'risk_level': self.disease_info[predicted_class]['risk'],
                'anomaly_score': float(anomaly_score),
                'confidence': 'High' if anomaly_score > 0.8 else 'Medium' if anomaly_score > 0.5 else 'Low'
            },
            'detailed_findings': {
                'chromosome_count': stats['total_chromosomes'],
                'count_status': 'Normal' if not stats.get('chromosome_count_abnormal', False) else 'Abnormal',
                'average_size': f"{stats['average_area']:.2f} pixels",
                'size_variation': f"{stats['size_variation']:.2%}",
                'aspect_ratio_range': f"{stats['aspect_ratio_stats']['min']:.2f} - {stats['aspect_ratio_stats']['max']:.2f}"
            },
            'possible_conditions': [],
            'recommendations': [],
            'risk_assessment': self.disease_info[predicted_class]['risk'],
            'next_steps': []
        }
        
        # Add possible diseases
        if predicted_class in [1, 2, 3, 4]:
            report['possible_conditions'] = self.disease_info[predicted_class]['diseases']
            
            # Add specific recommendations based on predicted class
            if predicted_class == 1:  # Chromosomal numerical abnormality
                report['recommendations'].extend([
                    'Perform chromosomal karyotyping to confirm aneuploidy',
                    'Consider amniocentesis or chorionic villus sampling',
                    'Consult genetic counselor for comprehensive evaluation'
                ])
            elif predicted_class == 2:  # Structural abnormality
                report['recommendations'].extend([
                    'Perform chromosomal microarray analysis (CMA)',
                    'Consider fluorescence in situ hybridization (FISH) for confirmation',
                    'Evaluate for possible microdeletion/microduplication syndromes'
                ])
        
        # Add general recommendations
        report['recommendations'].extend([
            'Consult clinical genetics specialist',
            'Comprehensive evaluation based on clinical symptoms',
            'Consider family member screening'
        ])
        
        # Next steps
        if anomaly_score > 0.7:
            report['next_steps'] = [
                'Schedule genetic counseling immediately',
                'Perform confirmatory genetic testing',
                'Develop monitoring and treatment plan'
            ]
            report['urgency'] = 'High'
        elif anomaly_score > 0.4:
            report['next_steps'] = [
                'Schedule genetic counseling appointment',
                'Consider further testing',
                'Regular follow-up'
            ]
            report['urgency'] = 'Medium'
        else:
            report['next_steps'] = ['Regular health check-ups']
            report['urgency'] = 'Low'
        
        # Technical notes
        report['technical_notes'] = {
            'analysis_method': 'Graph Convolutional Neural Network (GCN) + Image Processing',
            'model_version': '1.0',
            'confidence_threshold': 0.5,
            'limitations': 'AI-assisted diagnosis, requires clinical evaluation'
        }
        
        return report
    
    def save_report(self, report, output_dir='data/reports'):
        """Save report in multiple formats"""
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate file names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{report['patient_info'].get('patient_id', 'unknown')}_{timestamp}"
        
        # Save as JSON
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Save as text file
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        self.save_text_report(report, txt_path)
        
        # Try to save as PDF
        try:
            pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
            self.report_generator.save_report_as_pdf(report, pdf_path)
        except Exception as e:
            print(f"⚠️  PDF report generation failed: {e}")
            print("✅ JSON and text reports generated")
        
        print(f"✅ Report saved to: {output_dir}/")
        print(f"   - JSON: {os.path.basename(json_path)}")
        print(f"   - TXT: {os.path.basename(txt_path)}")
        
        return json_path, txt_path
    
    def save_text_report(self, report, filepath):
        """Save report in text format"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("          Genetic Image Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("【Patient Information】\n")
            f.write("-" * 40 + "\n")
            for key, value in report['patient_info'].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n【Analysis Summary】\n")
            f.write("-" * 40 + "\n")
            summary = report['analysis_summary']
            f.write(f"Analysis Result: {summary['result']}\n")
            f.write(f"Risk Level: {summary['risk_level']}\n")
            f.write(f"Anomaly Score: {summary['anomaly_score']:.3f}\n")
            f.write(f"Confidence: {summary['confidence']}\n")
            
            f.write("\n【Detailed Findings】\n")
            f.write("-" * 40 + "\n")
            findings = report['detailed_findings']
            for key, value in findings.items():
                f.write(f"{key}: {value}\n")
            
            if report['possible_conditions']:
                f.write("\n【Possible Conditions】\n")
                f.write("-" * 40 + "\n")
                for condition in report['possible_conditions']:
                    f.write(f"• {condition}\n")
            
            f.write("\n【Recommendations】\n")
            f.write("-" * 40 + "\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\n【Next Steps】\n")
            f.write("-" * 40 + "\n")
            for i, step in enumerate(report['next_steps'], 1):
                f.write(f"{i}. {step}\n")
            
            f.write(f"\nRisk Assessment: {report['risk_assessment']}\n")
            f.write(f"Urgency Level: {report.get('urgency', 'Low')}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Note: This report is generated by AI system, for reference only.\n")
            f.write("Final diagnosis must be confirmed by professional physician based on clinical data.\n")
            f.write("=" * 60 + "\n")
    
    def interactive_analysis(self):
        """Interactive analysis interface"""
        print("\n" + "="*60)
        print("       Interactive Genetic Image Analysis System")
        print("="*60)
        
        while True:
            print("\nPlease select operation:")
            print("1. Analyze single image")
            print("2. Batch analyze images")
            print("3. View example analysis")
            print("4. Exit system")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                self.analyze_single_image()
            elif choice == '2':
                self.analyze_batch_images()
            elif choice == '3':
                self.run_example()
            elif choice == '4':
                print("Thank you for using Genetic Image Analysis System!")
                break
            else:
                print("❌ Invalid choice, please re-enter")
    
    def analyze_single_image(self):
        """Analyze single image"""
        print("\n--- Single Image Analysis ---")
        
        # Ask for image path
        image_path = input("Enter image file path (or drag file here): ").strip().strip('"\'')
        
        if not os.path.exists(image_path):
            print(f"❌ File does not exist: {image_path}")
            return
        
        # Ask for patient information (optional)
        print("\n--- Patient Information (optional, press Enter to skip) ---")
        patient_info = {}
        patient_info['patient_id'] = input("Patient ID: ") or "Not provided"
        patient_info['age'] = input("Age: ") or "Not provided"
        patient_info['gender'] = input("Gender (M/F): ") or "Not provided"
        patient_info['clinical_notes'] = input("Clinical notes: ") or "None"
        
        # Start analysis
        result = self.process_image_file(image_path, patient_info)
        
        if result:
            # Save report
            report = result['report']
            json_path, txt_path = self.save_report(report)
            
            # Display summary
            print("\n✅ Analysis complete!")
            print(f"📋 Report generated:")
            print(f"   Result: {report['analysis_summary']['result']}")
            print(f"   Risk: {report['analysis_summary']['risk_level']}")
            print(f"   File: {os.path.basename(txt_path)}")
    
    def analyze_batch_images(self):
        """Batch analyze images"""
        print("\n--- Batch Image Analysis ---")
        
        folder_path = input("Enter image folder path: ").strip().strip('"\'')
        
        if not os.path.isdir(folder_path):
            print(f"❌ Folder does not exist: {folder_path}")
            return
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        image_files = []
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))
        
        if not image_files:
            print("❌ No image files found in folder")
            return
        
        print(f"Found {len(image_files)} image files")
        
        # Analyze each file
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Analyzing: {os.path.basename(image_path)}")
            
            # Create patient information
            patient_info = {
                'patient_id': f"BATCH_{i:03d}",
                'sample_id': os.path.basename(image_path),
                'batch_analysis': True
            }
            
            try:
                result = self.process_image_file(image_path, patient_info)
                if result:
                    self.save_report(result['report'])
            except Exception as e:
                print(f"❌ Analysis failed: {str(e)}")
    
    def run_example(self):
        """Run example analysis"""
        print("\n--- Example Analysis ---")
        
        # Check for example images
        example_dir = "data/examples"
        if not os.path.exists(example_dir):
            os.makedirs(example_dir, exist_ok=True)
            print(f"📁 Created example directory: {example_dir}")
            print("Please place test images in this directory and re-run")
            return
        
        # Find example images
        image_files = [f for f in os.listdir(example_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
        
        if not image_files:
            # Create a simulated image
            example_path = os.path.join(example_dir, "example_chromosomes.png")
            self.create_example_image(example_path)
            image_files = [example_path]
        
        # Analyze first example image
        image_path = os.path.join(example_dir, image_files[0])
        
        print(f"Using example image: {os.path.basename(image_path)}")
        
        # Example patient information
        patient_info = {
            'patient_id': 'EXAMPLE_001',
            'age': '32',
            'gender': 'Female',
            'clinical_notes': 'Routine prenatal check-up',
            'sample_type': 'Amniotic fluid cell chromosome karyotype'
        }
        
        # Analyze
        result = self.process_image_file(image_path, patient_info)
        
        if result:
            # Save report
            report = result['report']
            json_path, txt_path = self.save_report(report)
            
            # Display report summary
            print("\n" + "="*60)
            print("           Example Analysis Results Summary")
            print("="*60)
            print(f"Patient: {report['patient_info']['patient_id']}")
            print(f"Result: {report['analysis_summary']['result']}")
            print(f"Risk: {report['analysis_summary']['risk_level']}")
            print(f"Anomaly Score: {report['analysis_summary']['anomaly_score']:.3f}")
            
            if report['possible_conditions']:
                print(f"Possible Conditions: {', '.join(report['possible_conditions'][:3])}")
            
            print(f"\nDetailed report saved to: data/reports/")
            print("="*60)
    
    def create_example_image(self, output_path):
        """Create example chromosome image"""
        print("Creating example chromosome image...")
        
        # Create simulated image (normal karyotype)
        img = np.zeros((800, 1000), dtype=np.uint8)
        
        # Add 46 chromosomes (normal karyotype)
        positions = []
        for row in range(6):  # 6 rows
            for col in range(8):  # 8 per row, total 48 positions (some empty)
                if len(positions) >= 46:
                    break
                
                x = 100 + col * 120
                y = 100 + row * 120
                positions.append((x, y))
                
                # Draw chromosome (two chromatids)
                length = np.random.randint(40, 80)
                width = np.random.randint(10, 20)
                
                # Centromere position
                centromere = np.random.uniform(0.3, 0.7)
                
                # Draw left chromatid
                cv2.rectangle(img, (x-width//2, y), (x-width//2+width//2, y+int(length*centromere)), 200, -1)
                cv2.rectangle(img, (x-width//2, y+int(length*centromere)), (x-width//2+width//2, y+length), 200, -1)
                
                # Draw right chromatid
                cv2.rectangle(img, (x+1, y), (x+width//2, y+int(length*centromere)), 200, -1)
                cv2.rectangle(img, (x+1, y+int(length*centromere)), (x+width//2, y+length), 200, -1)
                
                # Draw centromere
                cv2.rectangle(img, (x-width//4, y+int(length*centromere)-2), 
                            (x+width//4, y+int(length*centromere)+2), 150, -1)
        
        # Save image
        cv2.imwrite(output_path, img)
        print(f"Example image created: {output_path}")

def main():
    """Main function"""
    print("Starting Genetic Image Analysis System...")
    
    # Create necessary directories
    directories = ['data', 'data/results', 'data/reports', 'data/models', 'data/examples']
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize system
    system = IntegratedMedicalSystem()
    
    # Check for model file
    model_path = 'data/models/medical_gcn_model.pth'
    if os.path.exists(model_path):
        system.load_model(model_path)
    else:
        print("ℹ️  No pre-trained model found, will use initialized model for analysis")
        print("   Recommended to train model with training data to improve accuracy")
    
    # Run interactive analysis
    system.interactive_analysis()

if __name__ == "__main__":
    main()