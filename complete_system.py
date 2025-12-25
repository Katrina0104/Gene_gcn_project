"""
Complete Genetic Image Analysis System - Single File Version
No separate files needed, all functionality is here
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Try importing optional packages, program can run even if not installed
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    print("⚠️  PyTorch not installed, will use traditional image analysis methods")
    HAS_TORCH = False

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    print("⚠️  FPDF not installed, PDF report functionality unavailable")
    HAS_FPDF = False

class ChromosomeAnalyzer:
    """Chromosome Analysis System"""
    
    def __init__(self):
        print("=" * 60)
        print("      Genetic Image Analysis System v2.0")
        print("=" * 60)
        
        # Disease database
        self.disease_database = {
            'trisomy_21': {
                'name': 'Down Syndrome',
                'chromosome': 'Chromosome 21',
                'abnormality': 'Trisomy 21',
                'symptoms': ['Intellectual disability', 'Characteristic facial features', 'Heart defects', 'Low muscle tone'],
                'prevalence': 'Approximately 1/700 newborns',
                'tests': ['Amniocentesis', 'Chorionic villus sampling', 'Non-invasive prenatal testing (NIPT)'],
                'management': ['Early intervention', 'Regular medical follow-up', 'Special education support']
            },
            'trisomy_18': {
                'name': 'Edwards Syndrome',
                'chromosome': 'Chromosome 18',
                'abnormality': 'Trisomy 18',
                'symptoms': ['Growth retardation', 'Multiple congenital anomalies', 'Severe intellectual disability'],
                'prevalence': 'Approximately 1/5000 newborns',
                'tests': ['Prenatal ultrasound', 'Chromosomal karyotyping'],
                'management': ['Supportive care', 'Symptomatic treatment']
            },
            'trisomy_13': {
                'name': 'Patau Syndrome',
                'chromosome': 'Chromosome 13',
                'abnormality': 'Trisomy 13',
                'symptoms': ['Severe intellectual disability', 'Cleft lip/palate', 'Polydactyly', 'Heart defects'],
                'prevalence': 'Approximately 1/10000 newborns',
                'tests': ['Prenatal diagnosis', 'Chromosomal analysis'],
                'management': ['Supportive therapy', 'Surgical correction of malformations']
            },
            'turner': {
                'name': 'Turner Syndrome',
                'chromosome': 'X Chromosome',
                'abnormality': 'Monosomy X',
                'symptoms': ['Short stature', 'Ovarian dysgenesis', 'Webbed neck', 'Heart abnormalities'],
                'prevalence': 'Approximately 1/2500 female infants',
                'tests': ['Chromosomal karyotyping'],
                'management': ['Growth hormone therapy', 'Estrogen replacement therapy', 'Regular follow-up']
            },
            'klinefelter': {
                'name': 'Klinefelter Syndrome',
                'chromosome': 'Sex Chromosomes',
                'abnormality': 'XXY',
                'symptoms': ['Small testes', 'Infertility', 'Gynecomastia', 'Learning difficulties'],
                'prevalence': 'Approximately 1/500-1000 male infants',
                'tests': ['Chromosomal karyotyping', 'Hormone testing'],
                'management': ['Testosterone replacement therapy', 'Fertility counseling', 'Educational support']
            },
            'cri_du_chat': {
                'name': 'Cri-du-chat Syndrome',
                'chromosome': 'Chromosome 5',
                'abnormality': 'Deletion',
                'symptoms': ['Cat-like cry', 'Microcephaly', 'Intellectual disability', 'Growth retardation'],
                'prevalence': 'Approximately 1/20000-50000 newborns',
                'tests': ['Chromosomal microarray analysis (CMA)'],
                'management': ['Early intervention', 'Speech therapy', 'Special education']
            }
        }
        
        # Create output directories
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories"""
        directories = ['data', 'data/results', 'data/reports', 'data/examples']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("📁 System directories created")
    
    def load_and_preprocess(self, image_path):
        """Load and preprocess image"""
        print(f"\n📷 Processing image: {os.path.basename(image_path)}")
        
        # Load image
        if not os.path.exists(image_path):
            print(f"❌ File does not exist: {image_path}")
            return None
        
        try:
            # Load image using PIL
            pil_image = Image.open(image_path)
            img_array = np.array(pil_image)
            
            # Convert to OpenCV format
            if len(img_array.shape) == 3:
                # If RGB, convert to BGR
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array
            
            print(f"✅ Image loaded successfully")
            print(f"   Dimensions: {img.shape}")
            print(f"   Type: {img.dtype}")
            
            return img
            
        except Exception as e:
            print(f"❌ Image loading failed: {e}")
            return None
    
    def detect_chromosomes(self, img):
        """Detect chromosomes"""
        print("\n🔬 Performing chromosome detection...")
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Image enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Binarization
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, hierarchy = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small noise and large background areas
        filtered_contours = []
        min_area = 50
        max_area = img.shape[0] * img.shape[1] * 0.5  # Maximum 50% of image area
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                filtered_contours.append(contour)
        
        print(f"✅ Detected {len(filtered_contours)} potential chromosomes")
        
        return {
            'original': img,
            'gray': gray,
            'enhanced': enhanced,
            'binary': binary,
            'morphed': morphed,
            'contours': filtered_contours
        }
    
    def extract_features(self, processing_results):
        """
        Extract chromosome features from contours / rotated boxes
        Compatible with NumPy >= 1.24
        """
        print("Extracting chromosome features...")

        img = processing_results["enhanced"]
        contours = processing_results["contours"]

        features = []
        boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # 過濾雜訊
                continue

            # 最小外接旋轉矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            # ✅ 修正點：不要用 np.int0
            box = box.astype(np.int32)

            boxes.append(box)

            # 幾何特徵
            width, height = rect[1]
            aspect_ratio = width / height if height != 0 else 0
            perimeter = cv2.arcLength(contour, True)

            # ROI
            x, y, w, h = cv2.boundingRect(contour)
            roi = img[y:y+h, x:x+w]

            if roi.size > 0:
                mean_intensity = np.mean(roi)
                std_intensity = np.std(roi)
            else:
                mean_intensity = 0
                std_intensity = 0

            features.append([
                area,
                perimeter,
                aspect_ratio,
                mean_intensity,
                std_intensity,
                w,
                h
            ])

        processing_results["features"] = np.array(features)
        processing_results["boxes"] = boxes

        print(f"✅ Extracted {len(features)} chromosome features")
        return processing_results
    
    def analyze_chromosome_pattern(self, features):
        """Analyze chromosome patterns"""
        print("\n🧬 Analyzing chromosome patterns...")

        if features is None or (hasattr(features, 'size') and features.size == 0) or len(features) == 0:
            print("No features detected")
            return {
                'chromosome_count': 0,
                'status': 'No chromosomes detected',
                'is_normal': False,
                'anomaly_score': 1.0,
                'suspected_conditions': ['Image quality issue or no chromosomes'],
                'confidence': 0.0
            }

        chromosome_count = len(features)

        # 將 features 轉成 list（如果是 numpy array）
        if isinstance(features, np.ndarray):
            features = features.tolist()

        # 根據 extract_features 的順序使用索引
        areas = [f[0] for f in features]
        perimeters = [f[1] for f in features]
        aspect_ratios = [f[2] for f in features]
        circularities = [4 * np.pi * f[0] / (f[1]**2) if f[1] != 0 else 0 for f in features]

        # Calculate statistics
        avg_area = np.mean(areas) if areas else 0
        std_area = np.std(areas) if len(areas) > 1 else 0
        cv_area = std_area / avg_area if avg_area > 0 else 0

        avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 0
        avg_circularity = np.mean(circularities) if circularities else 0

        # Initialize analysis results
        analysis = {
            'chromosome_count': chromosome_count,
            'statistics': {
                'average_area': float(avg_area),
                'area_variation': float(cv_area),
                'average_aspect_ratio': float(avg_aspect_ratio),
                'average_circularity': float(avg_circularity)
            }
        }

        # Analysis based on chromosome count
        anomaly_score = 0
        suspected_conditions = []

        if chromosome_count == 46:
            analysis['status'] = 'Normal chromosome count (46)'
            analysis['is_normal'] = True
            anomaly_score = 0.1
        elif chromosome_count == 47:
            analysis['status'] = 'Abnormal chromosome count (47, extra chromosome)'
            analysis['is_normal'] = False
            anomaly_score = 0.8
            if avg_aspect_ratio > 2.0:
                suspected_conditions.append('trisomy_21')  # Down Syndrome
            else:
                suspected_conditions.append('klinefelter')  # Klinefelter Syndrome
        elif chromosome_count == 45:
            analysis['status'] = 'Abnormal chromosome count (45, missing chromosome)'
            analysis['is_normal'] = False
            anomaly_score = 0.85
            suspected_conditions.append('turner')  # Turner Syndrome
        elif chromosome_count == 48:
            analysis['status'] = 'Severe chromosome count abnormality (48)'
            analysis['is_normal'] = False
            anomaly_score = 0.95
            suspected_conditions.append('Multiple chromosome abnormalities')
        elif chromosome_count < 45:
            analysis['status'] = f'Insufficient chromosome count ({chromosome_count})'
            analysis['is_normal'] = False
            anomaly_score = 0.9
            suspected_conditions.append('Severe chromosome deletion')
        elif chromosome_count > 47:
            analysis['status'] = f'Excessive chromosome count ({chromosome_count})'
            analysis['is_normal'] = False
            anomaly_score = 0.9
            suspected_conditions.append('Multiple chromosome duplication')
        else:
            analysis['status'] = f'Chromosome count: {chromosome_count}'
            analysis['is_normal'] = False
            anomaly_score = 0.5

        # Analysis based on shape variation
        if cv_area > 0.5:
            anomaly_score += 0.15
            analysis['status'] += ' + Size variation abnormality'
            suspected_conditions.append('Chromosome structural abnormality')

        if avg_circularity < 0.3:
            anomaly_score += 0.1
            analysis['status'] += ' + Shape abnormality'

        analysis['anomaly_score'] = min(float(anomaly_score), 1.0)
        analysis['suspected_conditions'] = suspected_conditions
        analysis['confidence'] = 1.0 - (std_area / avg_area if avg_area > 0 else 1.0)

        print(f"   Chromosome count: {chromosome_count}")
        print(f"   Status: {analysis['status']}")
        print(f"   Anomaly score: {analysis['anomaly_score']:.2f}")

        return analysis
    
    def generate_medical_report(self, analysis, patient_info=None, image_path=None):
        """Generate medical report"""
        print("\n📋 Generating medical report...")
        
        # Patient information
        if patient_info is None:
            patient_info = {
                'patient_id': 'Not provided',
                'patient_name': 'Not provided',
                'age': 'Not provided',
                'gender': 'Not provided',
                'sample_type': 'Chromosome karyotype',
                'referring_physician': 'Not provided',
                'collection_date': datetime.now().strftime('%Y-%m-%d')
            }
        
        # Report header
        report = {
            'report_header': {
                'title': 'Genetic Image Analysis Report',
                'report_id': f"GIA-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                'lab_name': 'AI Genetic Analysis Laboratory',
                'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'report_version': '2.0'
            },
            'patient_information': patient_info,
            'specimen_information': {
                'sample_id': patient_info.get('patient_id', 'Unknown'),
                'sample_type': 'Chromosome karyotype image',
                'image_filename': os.path.basename(image_path) if image_path else 'Unknown',
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            },
            'analysis_results': analysis,
            'clinical_interpretation': [],
            'differential_diagnosis': [],
            'recommendations': {
                'immediate': [],
                'follow_up': [],
                'counseling': []
            },
            'risk_assessment': {},
            'technical_notes': {
                'methodology': 'Computer vision + Pattern recognition analysis',
                'limitations': 'AI-assisted analysis, requires integration with clinical data',
                'disclaimer': 'This report is for reference only, not for final diagnosis'
            }
        }
        
        # Clinical interpretation
        if analysis['is_normal']:
            report['clinical_interpretation'].append({
                'finding': 'Normal chromosome count',
                'interpretation': 'No significant chromosome count abnormalities detected',
                'significance': 'Low risk'
            })
            report['risk_assessment']['overall_risk'] = 'Low'
            report['risk_assessment']['confidence'] = 'High'
        else:
            report['risk_assessment']['overall_risk'] = 'High' if analysis['anomaly_score'] > 0.7 else 'Medium'
            report['risk_assessment']['confidence'] = 'Medium-High' if analysis['confidence'] > 0.7 else 'Medium-Low'
            
            report['clinical_interpretation'].append({
                'finding': 'Chromosome abnormality detected',
                'interpretation': analysis['status'],
                'significance': 'Requires further confirmation'
            })
        
        # Differential diagnosis
        if analysis['suspected_conditions']:
            for condition_code in analysis['suspected_conditions']:
                if condition_code in self.disease_database:
                    disease = self.disease_database[condition_code]
                    report['differential_diagnosis'].append({
                        'condition': disease['name'],
                        'chromosome': disease['chromosome'],
                        'abnormality_type': disease['abnormality'],
                        'key_features': disease['symptoms'][:3],
                        'prevalence': disease['prevalence']
                    })
        
        # Recommendations
        if not analysis['is_normal']:
            report['recommendations']['immediate'].extend([
                'Immediate consultation with clinical genetics specialist',
                'Schedule genetic counseling appointment',
                'Perform confirmatory genetic testing'
            ])
            
            report['recommendations']['follow_up'].extend([
                'Regular chromosome follow-up examinations',
                'Detailed ultrasound evaluation',
                'Multidisciplinary team evaluation'
            ])
            
            report['recommendations']['counseling'].extend([
                'Genetic counseling and risk assessment',
                'Reproductive options counseling',
                'Family support and resource referral'
            ])
        else:
            report['recommendations']['follow_up'].extend([
                'Routine prenatal check-ups',
                'Consider advanced testing if concerns arise',
                'Maintain healthy lifestyle'
            ])
        
        print("✅ Report generation completed")
        return report
    
    def save_report_files(self, report, processing_results, output_dir='data/reports'):
        """Save report files"""
        print("\n💾 Saving report files...")
        
        # Create directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate base filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        patient_id = report['patient_information'].get('patient_id', 'unknown')
        base_filename = f"{patient_id}_{timestamp}"
        
        # 1. Save JSON report
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 2. Save text report
        txt_path = os.path.join(output_dir, f"{base_filename}.txt")
        self._save_text_report(report, txt_path)
        
        # 3. Save PDF report (if available)
        pdf_path = None
        if HAS_FPDF:
            pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
            self._save_pdf_report(report, pdf_path)
        
        # 4. Save visualization results
        viz_path = os.path.join('data/results', f"{base_filename}_visualization.png")
        self._save_visualization(processing_results, report, viz_path)
        
        print(f"✅ All files saved:")
        print(f"   📄 JSON report: {json_path}")
        print(f"   📄 Text report: {txt_path}")
        if pdf_path:
            print(f"   📄 PDF report: {pdf_path}")
        print(f"   🖼️  Visualization: {viz_path}")
        
        return {
            'json': json_path,
            'txt': txt_path,
            'pdf': pdf_path,
            'viz': viz_path
        }
    
    def _save_text_report(self, report, filepath):
        """Save report in text format"""
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 70 + "\n")
            f.write(" " * 20 + "Genetic Image Analysis Report\n")
            f.write("=" * 70 + "\n\n")
            
            # Report information
            f.write(f"Report ID: {report['report_header']['report_id']}\n")
            f.write(f"Generation Date: {report['report_header']['generation_date']}\n")
            f.write(f"Laboratory: {report['report_header']['lab_name']}\n")
            f.write("-" * 70 + "\n\n")
            
            # Patient information
            f.write("【Patient Information】\n")
            f.write("-" * 40 + "\n")
            for key, value in report['patient_information'].items():
                f.write(f"{key}: {value}\n")
            
            # Specimen information
            f.write("\n【Specimen Information】\n")
            f.write("-" * 40 + "\n")
            for key, value in report['specimen_information'].items():
                f.write(f"{key}: {value}\n")
            
            # Analysis results
            f.write("\n【Analysis Results】\n")
            f.write("-" * 40 + "\n")
            analysis = report['analysis_results']
            f.write(f"Chromosome Count: {analysis['chromosome_count']}\n")
            f.write(f"Detection Status: {analysis['status']}\n")
            f.write(f"Anomaly Score: {analysis['anomaly_score']:.3f}\n")
            f.write(f"Confidence: {analysis['confidence']:.2f}\n")
            f.write(f"Is Normal: {'Yes' if analysis['is_normal'] else 'No'}\n")
            
            # Statistical information
            if 'statistics' in analysis:
                f.write("\n【Statistical Information】\n")
                f.write("-" * 40 + "\n")
                for key, value in analysis['statistics'].items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.3f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            # Clinical interpretation
            if report['clinical_interpretation']:
                f.write("\n【Clinical Interpretation】\n")
                f.write("-" * 40 + "\n")
                for item in report['clinical_interpretation']:
                    f.write(f"Finding: {item['finding']}\n")
                    f.write(f"Interpretation: {item['interpretation']}\n")
                    f.write(f"Clinical Significance: {item['significance']}\n\n")
            
            # Differential diagnosis
            if report['differential_diagnosis']:
                f.write("\n【Differential Diagnosis】\n")
                f.write("-" * 40 + "\n")
                for i, item in enumerate(report['differential_diagnosis'], 1):
                    f.write(f"{i}. {item['condition']}\n")
                    f.write(f"   Chromosome: {item['chromosome']}\n")
                    f.write(f"   Abnormality Type: {item['abnormality_type']}\n")
                    f.write(f"   Prevalence: {item['prevalence']}\n")
            
            # Recommendations
            f.write("\n【Recommendations】\n")
            f.write("-" * 40 + "\n")
            
            if report['recommendations']['immediate']:
                f.write("Immediate Recommendations:\n")
                for i, rec in enumerate(report['recommendations']['immediate'], 1):
                    f.write(f"  {i}. {rec}\n")
            
            if report['recommendations']['follow_up']:
                f.write("\nFollow-up Recommendations:\n")
                for i, rec in enumerate(report['recommendations']['follow_up'], 1):
                    f.write(f"  {i}. {rec}\n")
            
            if report['recommendations']['counseling']:
                f.write("\nCounseling Recommendations:\n")
                for i, rec in enumerate(report['recommendations']['counseling'], 1):
                    f.write(f"  {i}. {rec}\n")
            
            # Risk assessment
            f.write("\n【Risk Assessment】\n")
            f.write("-" * 40 + "\n")
            for key, value in report['risk_assessment'].items():
                f.write(f"{key}: {value}\n")
            
            # Technical notes
            f.write("\n【Technical Notes】\n")
            f.write("-" * 40 + "\n")
            for key, value in report['technical_notes'].items():
                f.write(f"{key}: {value}\n")
            
            # Footer
            f.write("\n" + "=" * 70 + "\n")
            f.write("Important Notice:\n")
            f.write("1. This report is generated with AI system assistance\n")
            f.write("2. Results must be confirmed by professional genetic counselor\n")
            f.write("3. Please integrate with clinical symptoms for comprehensive judgment\n")
            f.write("4. Final diagnosis should be made by attending physician\n")
            f.write("=" * 70 + "\n")
    
    def _save_pdf_report(self, report, filepath):
        """Save PDF report"""
        if not HAS_FPDF:
            return
        
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Set font
            pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
            pdf.add_font('DejaVu', 'B', 'DejaVuSansCondensed-Bold.ttf', uni=True)
            
            # Title
            pdf.set_font('DejaVu', 'B', 16)
            pdf.cell(0, 10, 'Genetic Image Analysis Report', 0, 1, 'C')
            pdf.ln(5)
            
            # Report information
            pdf.set_font('DejaVu', '', 10)
            pdf.cell(0, 8, f"Report ID: {report['report_header']['report_id']}", 0, 1)
            pdf.cell(0, 8, f"Generation Date: {report['report_header']['generation_date']}", 0, 1)
            pdf.cell(0, 8, f"Laboratory: {report['report_header']['lab_name']}", 0, 1)
            pdf.ln(5)
            
            # Patient information
            pdf.set_font('DejaVu', 'B', 12)
            pdf.cell(0, 10, 'Patient Information', 0, 1)
            pdf.set_font('DejaVu', '', 10)
            for key, value in report['patient_information'].items():
                pdf.cell(0, 8, f"{key}: {value}", 0, 1)
            
            pdf.ln(5)
            
            # Analysis results
            pdf.set_font('DejaVu', 'B', 12)
            pdf.cell(0, 10, 'Analysis Results', 0, 1)
            pdf.set_font('DejaVu', '', 10)
            analysis = report['analysis_results']
            pdf.cell(0, 8, f"Chromosome Count: {analysis['chromosome_count']}", 0, 1)
            pdf.cell(0, 8, f"Detection Status: {analysis['status']}", 0, 1)
            pdf.cell(0, 8, f"Anomaly Score: {analysis['anomaly_score']:.3f}", 0, 1)
            pdf.cell(0, 8, f"Is Normal: {'Yes' if analysis['is_normal'] else 'No'}", 0, 1)
            
            # Recommendations
            pdf.ln(5)
            pdf.set_font('DejaVu', 'B', 12)
            pdf.cell(0, 10, 'Recommendations', 0, 1)
            pdf.set_font('DejaVu', '', 10)
            
            for category, recommendations in report['recommendations'].items():
                if recommendations:
                    pdf.cell(0, 8, f"{category}:", 0, 1)
                    for rec in recommendations:
                        pdf.cell(10)  # Indentation
                        pdf.cell(0, 8, f"• {rec}", 0, 1)
                    pdf.ln(2)
            
            # Footer
            pdf.ln(10)
            pdf.set_font('DejaVu', 'I', 8)
            pdf.multi_cell(0, 5, 'Note: This report is generated with AI system assistance, for reference only. Final diagnosis should be confirmed by professional physician.')
            
            pdf.output(filepath)
            
        except Exception as e:
            print(f"⚠️  PDF report generation failed: {e}")
    
    def _save_visualization(self, processing_results, report, filepath):
        """Save visualization results"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            img = processing_results['original']
            enhanced = processing_results['enhanced']
            binary = processing_results['binary']
            contours = processing_results['contours']
            analysis = report['analysis_results']
            
            # Original image
            if len(img.shape) == 3:
                axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                axes[0, 0].imshow(img, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Enhanced image
            axes[0, 1].imshow(enhanced, cmap='gray')
            axes[0, 1].set_title('Enhanced Image')
            axes[0, 1].axis('off')
            
            # Binary image
            axes[0, 2].imshow(binary, cmap='gray')
            axes[0, 2].set_title('Binary Image')
            axes[0, 2].axis('off')
            
            # Contour detection
            contour_img = img.copy()
            if len(contour_img.shape) == 2:
                contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            axes[1, 0].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title(f'Chromosome Contours ({len(contours)} detected)')
            axes[1, 0].axis('off')
            
            # Chromosome count chart
            chromosome_count = analysis['chromosome_count']
            normal_count = 46
            colors = ['green' if chromosome_count == normal_count else 'red']
            axes[1, 1].bar(['Detected Count', 'Normal Count'], 
                          [chromosome_count, normal_count], 
                          color=colors + ['blue'])
            axes[1, 1].set_title('Chromosome Count Comparison')
            axes[1, 1].set_ylabel('Count')
            
            # Anomaly score
            anomaly_score = analysis['anomaly_score']
            ax = axes[1, 2]
            ax.barh(['Anomaly Score'], [anomaly_score], color='red' if anomaly_score > 0.5 else 'green')
            ax.set_xlim(0, 1)
            ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)
            ax.set_title(f'Anomaly Score: {anomaly_score:.3f}')
            
            # Overall title
            status = analysis['status']
            fig.suptitle(f'Genetic Image Analysis Results - {status}', fontsize=16, y=0.98)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Visualization save failed: {e}")
    
    def create_sample_image(self):
        """Create sample image"""
        print("\n🎨 Creating sample chromosome image...")
        
        # Create an 800x1000 image
        img = np.zeros((800, 1000, 3), dtype=np.uint8)
        img.fill(240)  # Light gray background
        
        # Normal human has 46 chromosomes
        positions = []
        chromosome_types = ['metacentric', 'submetacentric', 'acrocentric']
        
        for i in range(46):
            # Random position (avoid overlap)
            while True:
                x = np.random.randint(100, 900)
                y = np.random.randint(100, 700)
                if all(np.sqrt((x-p[0])**2 + (y-p[1])**2) > 80 for p in positions):
                    positions.append((x, y))
                    break
            
            # Random chromosome type
            chrom_type = np.random.choice(chromosome_types)
            
            # Set dimensions based on type
            if chrom_type == 'metacentric':
                length = np.random.randint(60, 100)
                arm_ratio = 0.5  # Centromere in middle
            elif chrom_type == 'submetacentric':
                length = np.random.randint(50, 90)
                arm_ratio = np.random.uniform(0.3, 0.4)  # Centromere off-center
            else:  # acrocentric
                length = np.random.randint(40, 70)
                arm_ratio = 0.2  # Centromere near end
            
            width = np.random.randint(15, 25)
            
            # Color (different colors for different chromosome groups)
            if i < 22:  # Autosomes
                color = (100, 100, 200)  # Blue
            elif i < 44:  # Another group of autosomes
                color = (200, 100, 100)  # Red
            else:  # Sex chromosomes
                color = (100, 200, 100)  # Green
            
            # Draw chromosome (two chromatids)
            # Left chromatid
            cv2.rectangle(img, 
                         (x - width//2, y), 
                         (x - width//2 + width//2, y + int(length * arm_ratio)), 
                         color, -1)
            cv2.rectangle(img, 
                         (x - width//2, y + int(length * arm_ratio)), 
                         (x - width//2 + width//2, y + length), 
                         color, -1)
            
            # Right chromatid
            cv2.rectangle(img, 
                         (x + 1, y), 
                         (x + width//2, y + int(length * arm_ratio)), 
                         color, -1)
            cv2.rectangle(img, 
                         (x + 1, y + int(length * arm_ratio)), 
                         (x + width//2, y + length), 
                         color, -1)
            
            # Draw centromere
            cv2.rectangle(img, 
                         (x - width//4, y + int(length * arm_ratio) - 3), 
                         (x + width//4, y + int(length * arm_ratio) + 3), 
                         (50, 50, 50), -1)
        
        # Save image
        sample_path = 'data/examples/sample_chromosomes.jpg'
        cv2.imwrite(sample_path, img)
        print(f"✅ Sample image created: {sample_path}")
        
        return sample_path

    # ===== GUI Function Methods =====
    
    def select_image_gui(self):
        """Use Tkinter GUI to select image"""
        import tkinter as tk
        from tkinter import filedialog, simpledialog
        
        # Create root window and hide it
        root = tk.Tk()
        root.withdraw()  # Hide main window
        root.attributes('-topmost', True)  # Ensure window is on top
        
        # Set up file selection dialog
        file_path = filedialog.askopenfilename(
            title="Select Genetic/Chromosome Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()  # Destroy window
        
        if not file_path:
            print("User cancelled selection")
            return None
        
        # Ask for patient information (optional)
        patient_info = self._ask_patient_info_gui()
        
        return file_path, patient_info
    
    def _ask_patient_info_gui(self):
        """Ask for patient information through dialog"""
        import tkinter as tk
        from tkinter import simpledialog
        
        info = {}
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Use simple input dialog
        info['patient_id'] = simpledialog.askstring("Patient Information", "Patient ID (optional):", parent=root)
        info['patient_name'] = simpledialog.askstring("Patient Information", "Patient Name (optional):", parent=root)
        info['age'] = simpledialog.askstring("Patient Information", "Age (optional):", parent=root)
        info['gender'] = simpledialog.askstring("Patient Information", "Gender (M/F, optional):", parent=root)
        
        root.destroy()
        
        # Handle empty inputs
        return {k: (v if v else "Not provided") for k, v in info.items()}
    
    def analyze_single_image_gui(self):
        """Analyze single image through GUI"""
        print("\n" + "="*60)
        print("Graphical Interface Analysis Mode")
        print("="*60)
        
        # Select file
        result = self.select_image_gui()
        if not result:
            print("Analysis cancelled.")
            return
        
        image_path, patient_info = result
        
        print(f"📂 Selected file: {os.path.basename(image_path)}")
        print("⏳ Starting analysis...")
        
        # Perform analysis (call your existing analysis pipeline)
        try:
            img = self.load_and_preprocess(image_path)
            if img is None:
                return
            
            processing_results = self.detect_chromosomes(img)
            processing_results = self.extract_features(processing_results)
            analysis = self.analyze_chromosome_pattern(processing_results['features'])
            report = self.generate_medical_report(analysis, patient_info, image_path)
            saved_files = self.save_report_files(report, processing_results)
            
            # Display result summary
            self._show_results_gui(analysis, saved_files)
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_results_gui(self, analysis, saved_files):
        """Display result summary in GUI"""
        import tkinter as tk
        from tkinter import scrolledtext
        
        root = tk.Tk()
        root.title("✅ Analysis Results Summary")
        root.geometry("550x450")
        
        # Main frame
        main_frame = tk.Frame(root, padx=20, pady=20)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Title
        title_label = tk.Label(main_frame, text="Genetic Image Analysis Complete", 
                               font=("Arial", 16, "bold"), fg="#2c3e50")
        title_label.pack(pady=(0, 15))
        
        # Create text box to display results (with scrollbar)
        text_frame = tk.Frame(main_frame)
        text_frame.pack(expand=True, fill=tk.BOTH)
        
        text_widget = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, 
                                                font=("Consolas", 10), 
                                                height=15, width=60)
        text_widget.pack(expand=True, fill=tk.BOTH)
        
        # Add result information
        risk_level = 'High' if analysis['anomaly_score'] > 0.7 else 'Medium' if analysis['anomaly_score'] > 0.3 else 'Low'
        
        result_text = f"""【Analysis Results Summary】
{'='*55}
📊 Chromosome Count: {analysis['chromosome_count']}
📈 Detection Status: {analysis['status']}
⚠️  Anomaly Score: {analysis['anomaly_score']:.3f}
✅ Is Normal: {'Yes' if analysis['is_normal'] else 'No'}
🎯 Risk Level: {risk_level}

【Report Files Saved】
📄 Text Report: {os.path.basename(saved_files.get('txt', ''))}
"""
        
        if saved_files.get('pdf'):
            result_text += f"📄 PDF Report: {os.path.basename(saved_files['pdf'])}\n"
        
        if saved_files.get('viz'):
            result_text += f"🖼️  Analysis Chart: {os.path.basename(saved_files['viz'])}"
        
        text_widget.insert(tk.END, result_text)
        text_widget.config(state=tk.DISABLED)  # Set to read-only
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=15)
        
        # OK button
        tk.Button(button_frame, text="Done", command=root.destroy, 
                 bg="#27ae60", fg="white", font=("Arial", 10, "bold"),
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        # View reports button (optional)
        def open_report_dir():
            import os, platform, subprocess
            report_dir = os.path.dirname(saved_files.get('txt', ''))
            if os.path.exists(report_dir):
                if platform.system() == "Windows":
                    os.startfile(report_dir)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen(["open", report_dir])
                else:  # Linux
                    subprocess.Popen(["xdg-open", report_dir])
        
        tk.Button(button_frame, text="Open Reports Folder", command=open_report_dir,
                 bg="#3498db", fg="white", font=("Arial", 10),
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        root.mainloop()

    # ===== Command Line Function Methods =====
    
    def run_interactive_analysis(self):
        """Interactive analysis interface"""
        print("\n" + "="*60)
        print("           Interactive Genetic Image Analysis")
        print("="*60)
        
        while True:
            print("\nPlease select operation:")
            print("1. 📷 Analyze single image file")
            print("2. 📂 Analyze multiple images in folder")
            print("3. 🎨 Create and analyze sample image")
            print("4. 📊 View disease database")
            print("5. ❌ Exit system")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                self.analyze_single_image_cli()
            elif choice == '2':
                self.analyze_batch_images()
            elif choice == '3':
                self.create_and_analyze_sample_cli()
            elif choice == '4':
                self.show_disease_database()
            elif choice == '5':
                print("\nThank you for using Genetic Image Analysis System!")
                print("Goodbye! 👋")
                break
            else:
                print("❌ Invalid choice, please re-enter")
    
    def analyze_single_image_cli(self):
        """Command line analysis of single image"""
        print("\n" + "="*40)
        print("Single Image Analysis")
        print("="*40)
        
        # Input image path
        image_path = input("Enter image file path: ").strip().strip('"\'')
        
        if not os.path.exists(image_path):
            print(f"❌ File does not exist: {image_path}")
            return
        
        # Input patient information
        print("\n--- Patient Information (press Enter to skip) ---")
        patient_info = {}
        patient_info['patient_id'] = input("Patient ID: ") or "PAT001"
        patient_info['patient_name'] = input("Patient Name: ") or "Not provided"
        patient_info['age'] = input("Age: ") or "Not provided"
        patient_info['gender'] = input("Gender (M/F): ") or "Not provided"
        patient_info['sample_type'] = input("Sample Type: ") or "Chromosome karyotype"
        patient_info['clinical_notes'] = input("Clinical Notes: ") or "None"
        
        print(f"\nStarting analysis: {os.path.basename(image_path)}")
        print("-" * 40)
        
        try:
            # 1. Load image
            img = self.load_and_preprocess(image_path)
            if img is None:
                return
            
            # 2. Detect chromosomes
            processing_results = self.detect_chromosomes(img)
            
            # 3. Extract features
            processing_results = self.extract_features(processing_results)
            
            # 4. Analyze patterns
            analysis = self.analyze_chromosome_pattern(processing_results['features'])
            
            # 5. Generate report
            report = self.generate_medical_report(analysis, patient_info, image_path)
            
            # 6. Save report
            saved_files = self.save_report_files(report, processing_results)
            
            # 7. Display summary
            print("\n" + "="*40)
            print("Analysis Complete!")
            print("="*40)
            print(f"📊 Result Summary:")
            print(f"   Chromosome Count: {analysis['chromosome_count']}")
            print(f"   Detection Status: {analysis['status']}")
            print(f"   Anomaly Score: {analysis['anomaly_score']:.3f}")
            print(f"   Risk Level: {'High' if analysis['anomaly_score'] > 0.7 else 'Medium' if analysis['anomaly_score'] > 0.3 else 'Low'}")
            
            if analysis['suspected_conditions']:
                print(f"   Possible Conditions: {', '.join([self.disease_database.get(c, {}).get('name', c) for c in analysis['suspected_conditions']])}")
            
            print(f"\n📁 Report Files:")
            print(f"   📄 {os.path.basename(saved_files['txt'])}")
            if saved_files.get('pdf'):
                print(f"   📄 {os.path.basename(saved_files['pdf'])}")
            print(f"   🖼️  {os.path.basename(saved_files['viz'])}")
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_batch_images(self):
        """Batch analyze images"""
        print("\n" + "="*40)
        print("Batch Image Analysis")
        print("="*40)
        
        folder_path = input("Enter image folder path: ").strip().strip('"\'')
        
        if not os.path.isdir(folder_path):
            print(f"❌ Folder does not exist: {folder_path}")
            return
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))
        
        if not image_files:
            print("❌ No supported image files found in folder")
            return
        
        print(f"Found {len(image_files)} image files")
        
        # Analyze each file
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Analyzing: {os.path.basename(image_path)}")
            print("-" * 30)
            
            try:
                # Auto-generate patient information
                patient_info = {
                    'patient_id': f"BATCH_{i:03d}",
                    'patient_name': f"Batch Analysis_{i}",
                    'sample_type': 'Chromosome karyotype',
                    'batch_analysis': True
                }
                
                # Load image
                img = self.load_and_preprocess(image_path)
                if img is None:
                    continue
                
                # Detect chromosomes
                processing_results = self.detect_chromosomes(img)
                
                # Extract features
                processing_results = self.extract_features(processing_results)
                
                # Analyze patterns
                analysis = self.analyze_chromosome_pattern(processing_results['features'])
                
                # Generate report
                report = self.generate_medical_report(analysis, patient_info, image_path)
                
                # Save report
                self.save_report_files(report, processing_results)
                
                print(f"✅ Completed")
                
            except Exception as e:
                print(f"❌ Analysis failed: {e}")
        
        print(f"\n✅ Batch analysis complete! Processed {len(image_files)} files")
        print(f"📁 All reports saved to: data/reports/")
    
    def create_and_analyze_sample_cli(self):
        """Create and analyze sample image (command line version)"""
        print("\n" + "="*40)
        print("Sample Image Analysis")
        print("="*40)
        
        # Create sample image
        sample_path = self.create_sample_image()
        
        # Analyze sample image
        patient_info = {
            'patient_id': 'SAMPLE_001',
            'patient_name': 'Sample Patient',
            'age': '32',
            'gender': 'Female',
            'sample_type': 'Chromosome karyotype (Sample)',
            'clinical_notes': 'Routine prenatal check-up - Sample analysis'
        }
        
        print("\nStarting sample image analysis...")
        print("-" * 30)
        
        try:
            # Load image
            img = self.load_and_preprocess(sample_path)
            
            # Detect chromosomes
            processing_results = self.detect_chromosomes(img)
            
            # Extract features
            processing_results = self.extract_features(processing_results)
            
            # Analyze patterns
            analysis = self.analyze_chromosome_pattern(processing_results['features'])
            
            # Generate report
            report = self.generate_medical_report(analysis, patient_info, sample_path)
            
            # Save report
            saved_files = self.save_report_files(report, processing_results)
            
            print("\n✅ Sample analysis complete!")
            print(f"📁 Report files: data/reports/")
            
            # Display results
            print(f"\n📊 Sample Analysis Results:")
            print(f"   Chromosome Count: {analysis['chromosome_count']} (Expected: 46)")
            print(f"   Detection Status: {analysis['status']}")
            print(f"   Demonstration Purpose: System functionality showcase")
            
        except Exception as e:
            print(f"❌ Sample analysis failed: {e}")
    
    def show_disease_database(self):
        """Display disease database"""
        print("\n" + "="*60)
        print("Disease Database")
        print("="*60)
        
        for code, disease in self.disease_database.items():
            print(f"\n🏥 {disease['name']}")
            print(f"  Chromosome: {disease['chromosome']}")
            print(f"  Abnormality Type: {disease['abnormality']}")
            print(f"  Prevalence: {disease['prevalence']}")
            print(f"  Main Symptoms: {', '.join(disease['symptoms'][:3])}")
            print(f"  Testing Methods: {', '.join(disease['tests'])}")
            print(f"  Management: {', '.join(disease['management'][:2])}")
        
        input("\nPress Enter to continue...")

# Main program
def main():
    """Main function"""
    print("="*60)
    print("      Genetic Image Analysis System v2.0 - GUI Version")
    print("="*60)
    
    # Create analyzer instance
    analyzer = ChromosomeAnalyzer()
    
    while True:
        print("\nPlease select operation mode:")
        print("1. 🖱️  Graphical Interface Mode (Recommended - Click to select files)")
        print("2. ⌨️  Command Line Interactive Mode (Manual path input)")
        print("3. 🎨 Create and analyze sample image")
        print("4. 📊 View disease database")
        print("5. ❌ Exit system")
        
        choice = input("\nSelect (1-5): ").strip()
        
        if choice == '1':
            # Graphical interface mode
            analyzer.analyze_single_image_gui()
            # Ask if continue after analysis
            continue_choice = input("\nContinue analyzing other images? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\nThank you for using Genetic Image Analysis System!")
                break
                
        elif choice == '2':
            # Command line mode
            analyzer.run_interactive_analysis()
            break
            
        elif choice == '3':
            # Sample mode
            analyzer.create_and_analyze_sample_cli()
            
            # Ask if continue
            continue_choice = input("\nReturn to main menu? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\nThank you for using Genetic Image Analysis System!")
                break
                
        elif choice == '4':
            # View disease database
            analyzer.show_disease_database()
            
        elif choice == '5':
            print("\nThank you for using Genetic Image Analysis System!")
            break
            
        else:
            print("❌ Invalid selection, please re-enter")

if __name__ == "__main__":
    # Ensure program can start normally
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted.")
    except Exception as e:
        print(f"\n❌ Program startup error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Please check:")
        print("1. Required packages installed: pip install opencv-python matplotlib pillow numpy")
        print("2. Python version compatibility")
        print("3. File permission issues")
    input("\nPress Enter to exit...")