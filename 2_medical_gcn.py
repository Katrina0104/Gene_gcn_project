# 2_medical_gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torchvision.models as models
import numpy as np

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
                'analysis_date': image_info.get('date', '2024-01-01')
            },
            'analysis_results': {
                'abnormality_detected': anomaly_scores > 0.5,
                'anomaly_score': float(anomaly_scores),
                'predicted_class': torch.argmax(predictions).item(),
                'class_probabilities': F.softmax(predictions, dim=1).tolist()
            },
            'detected_abnormalities': [],
            'recommendations': [],
            'risk_assessment': 'Low Risk'
        }
        
        # Add detected abnormalities based on prediction results
        if anomaly_scores > 0.7:
            predicted_class = torch.argmax(predictions).item()
            
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
            
            pdf.output(filename)
            print(f"✅ Report saved as: {filename}")
            
        except ImportError:
            print("❌ Requires fpdf installation: pip install fpdf")
            # Save as text file
            with open(filename.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
                import json
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"✅ Report saved as text file: {filename.replace('.pdf', '.txt')}")

def test_model():
    print("=== Testing Medical Image GCN Model ===")
    
    # Create test data
    num_nodes = 46  # 46 chromosomes
    num_features = 7  # 7 features
    
    x = torch.randn((num_nodes, num_features))
    edge_index = torch.tensor([[i, i+1] for i in range(num_nodes-1)] + 
                              [[i+1, i] for i in range(num_nodes-1)], dtype=torch.long).t().contiguous()
    
    # Create model
    model = MedicalImageGCN(num_features=num_features)
    
    # Test forward propagation
    class_predictions, anomaly_scores = model(x, edge_index)
    
    print(f"✅ Model test successful!")
    print(f"   Input features: {x.shape}")
    print(f"   Edge index: {edge_index.shape}")
    print(f"   Classification output: {class_predictions.shape}")
    print(f"   Anomaly scores: {anomaly_scores.shape}")
    
    # Test report generation
    report_gen = MedicalReportGenerator()
    report = report_gen.generate_report(
        predictions=class_predictions[0].unsqueeze(0),
        anomaly_scores=anomaly_scores[0],
        image_info={'sample_id': 'TEST001', 'image_type': 'Chromosome Karyotype'}
    )
    
    print(f"\nGenerated report summary:")
    print(f"  Abnormality detected: {report['analysis_results']['abnormality_detected']}")
    print(f"  Anomaly score: {report['analysis_results']['anomaly_score']:.3f}")
    print(f"  Risk assessment: {report['risk_assessment']}")
    
    # Save report
    report_gen.save_report_as_pdf(report, 'data/medical_report.pdf')
    
    return model

if __name__ == "__main__":
    model = test_model()