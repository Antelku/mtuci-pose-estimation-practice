import cv2
import numpy as np
import json
import time
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp
import os
import glob


from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont




def register_fonts():
    font_files = [
        ('C:\\Windows\\Fonts\\arial.ttf', 'Arial'),
        ('C:\\Windows\\Fonts\\arialbd.ttf', 'Arial-Bold'),
    ]
    registered = False
    for font_path, font_name in font_files:
        try:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                registered = True
        except Exception as e:
            print(f"  Ошибка: {e}")
    return registered

print("Регистрация шрифтов...")
FONTS_AVAILABLE = register_fonts()
MAIN_FONT = 'Arial' if FONTS_AVAILABLE else 'Helvetica'
BOLD_FONT = 'Arial-Bold' if FONTS_AVAILABLE else 'Helvetica-Bold'




print("\nЗагрузка моделей...")
yolo_model = YOLO('yolov8n-pose.pt')
print("  YOLOv8-Pose загружена")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
media_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5
)
print("  MediaPipe загружена")

ARCHITECTURES = {
    'YOLOv8-Pose': {'fps': None, 'map': 0.506, 'size': '6.5 МБ', 'approach': 'One-stage'},
    'MediaPipe':   {'fps': None, 'map': 0.650, 'size': '12 МБ',  'approach': 'Top-down'},
    'HRNet-W48':   {'fps': 12.0, 'map': 0.765, 'size': '250 МБ','approach': 'Top-down'},
    'OpenPose':    {'fps': 8.0,  'map': 0.685, 'size': '200 МБ','approach': 'Bottom-up'},
    'RTMPose-L':   {'fps': 65.0, 'map': 0.742, 'size': '45 МБ', 'approach': 'Top-down'},
}




def calculate_angle(p1, p2, p3):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    a = p1 - p2
    b = p3 - p2
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))




def analyze_squat(keypoints):
    errors = []
    angles = {}
    hip_l, knee_l, ankle_l = keypoints[11], keypoints[13], keypoints[15]
    hip_r, knee_r, ankle_r = keypoints[12], keypoints[14], keypoints[16]
    shoulder_l, shoulder_r = keypoints[5], keypoints[6]
    
    knee_l_ang = calculate_angle(hip_l, knee_l, ankle_l)
    knee_r_ang = calculate_angle(hip_r, knee_r, ankle_r)
    angles['knee_left'] = knee_l_ang
    angles['knee_right'] = knee_r_ang
    
    hip_c = (hip_l + hip_r) / 2
    sh_c = (shoulder_l + shoulder_r) / 2
    torso = calculate_angle(sh_c, hip_c, knee_l)
    angles['torso'] = torso
    
    avg_knee = (knee_l_ang + knee_r_ang) / 2
    if avg_knee < 60:
        errors.append("Слишком глубокий присед (риск для коленей)")
    elif avg_knee > 110:
        errors.append("Недостаточная глубина приседа")
    elif 70 <= avg_knee <= 100:
        errors.append("Хорошая глубина!")
    
    if torso < 60:
        errors.append("Слишком сильный наклон вперёд")
    if torso > 110:
        errors.append("Спина слишком прямая")
    if knee_l[0] > ankle_l[0] + 50 or knee_r[0] > ankle_r[0] + 50:
        errors.append("Колени выходят за носки")
    if abs(knee_l_ang - knee_r_ang) > 20:
        errors.append("Асимметрия в коленях")
    
    return angles, errors




def process_image_yolo(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, {'error': 'Failed to load image'}
    
    results = yolo_model(img, verbose=False)
    annotated = results[0].plot()
    
    if results[0].keypoints is not None:
        kpts = results[0].keypoints[0].xy[0].cpu().numpy()
        conf = results[0].keypoints[0].conf[0].cpu().numpy()
        
        if len(kpts) >= 17:
            angles, errors = analyze_squat(kpts)
            
            cv2.putText(annotated, "SQUAT", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            return annotated, {
                'angles': angles,
                'errors': errors,
                'confidence': float(np.mean(conf)),
                'framework': 'YOLOv8-Pose'
            }
    
    return annotated, {'error': 'No pose detected'}

def process_image_mediapipe(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, {'error': 'Failed to load image'}
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = media_model.process(img_rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        landmarks = results.pose_landmarks.landmark
        h, w = img.shape[:2]
        keypoints = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in range(17)])
        
        angles, errors = analyze_squat(keypoints)
        
        cv2.putText(img, "SQUAT", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return img, {
            'angles': angles,
            'errors': errors,
            'confidence': 0.95,
            'framework': 'MediaPipe'
        }
    
    return img, {'error': 'No pose detected'}




def benchmark_all(image_path, iterations=10):
    print("\nЗамер скорости архитектур...")
    benchmarks = {}
    
    print("  YOLOv8-Pose...", end=' ')
    for _ in range(3):
        process_image_yolo(image_path)
    start = time.time()
    for _ in range(iterations):
        process_image_yolo(image_path)
    fps = iterations / (time.time() - start)
    benchmarks['YOLOv8-Pose'] = {'fps': round(fps, 2)}
    print(f"{fps:.2f} FPS")
    
    print("  MediaPipe...", end=' ')
    for _ in range(3):
        process_image_mediapipe(image_path)
    start = time.time()
    for _ in range(iterations):
        process_image_mediapipe(image_path)
    fps = iterations / (time.time() - start)
    benchmarks['MediaPipe'] = {'fps': round(fps, 2)}
    print(f"{fps:.2f} FPS")
    
    for name in ['HRNet-W48', 'OpenPose', 'RTMPose-L']:
        benchmarks[name] = {'fps': ARCHITECTURES[name]['fps']}
        print(f"  {name}: {ARCHITECTURES[name]['fps']} FPS (официальные данные)")
    
    return benchmarks




def add_page_number(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont(MAIN_FONT, 9)
    page_num = canvas_obj.getPageNumber()
    canvas_obj.drawCentredString(A4[0] / 2.0, 1.5 * cm, f"Страница {page_num}")
    canvas_obj.drawRightString(A4[0] - 2 * cm, 1.5 * cm, "Squat Analysis Report")
    canvas_obj.restoreState()

def generate_pdf_report(stats, output_path='results/report.pdf'):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                  fontName=BOLD_FONT, fontSize=20,
                                  alignment=TA_CENTER, spaceAfter=20)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                    fontName=BOLD_FONT, fontSize=14,
                                    spaceBefore=15, spaceAfter=10)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontName=MAIN_FONT, fontSize=11, spaceAfter=6)
    error_style = ParagraphStyle('Error', parent=styles['Normal'],
                                  fontName=MAIN_FONT, fontSize=10,
                                  textColor=colors.red, leftIndent=20)
    ok_style = ParagraphStyle('OK', parent=styles['Normal'],
                               fontName=MAIN_FONT, fontSize=10,
                               textColor=colors.green, leftIndent=20)
    
    elements = []
    elements.append(Paragraph("Анализ техники приседаний", title_style))
    elements.append(Spacer(1, 0.5 * cm))
    
    elements.append(Paragraph("Общая информация", heading_style))
    
    info_data = [
        ['Дата генерации:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Обработано изображений:', str(stats['total_images'])],
        ['Тип упражнения:', 'Приседание'],
        ['Количество архитектур:', '5'],
    ]
    info_table = Table(info_data, colWidths=[5*cm, 9*cm])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), MAIN_FONT),
        ('FONTNAME', (0, 0), (0, -1), BOLD_FONT),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.5 * cm))
    
    elements.append(Paragraph("Сравнение 5 архитектур Pose Estimation", heading_style))
    arch_data = [['Архитектура', 'Подход', 'mAP', 'FPS', 'Размер']]
    for name in ['YOLOv8-Pose', 'MediaPipe', 'HRNet-W48', 'OpenPose', 'RTMPose-L']:
        arch_data.append([
            name,
            ARCHITECTURES[name]['approach'],
            str(ARCHITECTURES[name]['map']),
            f"{stats['benchmarks'][name]['fps']}",
            ARCHITECTURES[name]['size']
        ])
    
    arch_table = Table(arch_data, colWidths=[3*cm, 2*cm, 1.5*cm, 1.5*cm, 2*cm])
    arch_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), MAIN_FONT),
        ('FONTNAME', (0, 0), (-1, 0), BOLD_FONT),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(arch_table)
    elements.append(Spacer(1, 0.5 * cm))
    
    elements.append(Paragraph("Результаты анализа", heading_style))
    results = stats.get('results', [])
    
    for i, result in enumerate(results, 1):
        img_name = os.path.basename(result.get('image', 'unknown'))
        
        elements.append(Paragraph(f"<b>#{i}. {img_name}</b>", body_style))
        
        angles = result.get('angles', {})
        if angles:
            angles_data = [['Сустав', 'Угол (градусы)']]
            for angle_name, angle_value in angles.items():
                joint_name = angle_name.replace('_', ' ').title()
                angles_data.append([joint_name, f"{angle_value:.1f}°"])
            
            angles_table = Table(angles_data, colWidths=[6*cm, 4*cm])
            angles_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), MAIN_FONT),
                ('FONTNAME', (0, 0), (-1, 0), BOLD_FONT),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            elements.append(angles_table)
            elements.append(Spacer(1, 0.2 * cm))
        
        errors = result.get('errors', [])
        if errors:
            elements.append(Paragraph(f"<b>Обнаружено ошибок: {len(errors)}</b>", error_style))
            for err in errors:
                elements.append(Paragraph(f"• {err}", error_style))
        else:
            elements.append(Paragraph("✓ Техника: корректна", ok_style))
        
        conf = result.get('confidence')
        if conf is not None:
            elements.append(Paragraph(f"Уверенность: {conf:.3f}", body_style))
        
        elements.append(Spacer(1, 0.3 * cm))
        if i % 4 == 0 and i < len(results):
            elements.append(PageBreak())
    
    elements.append(PageBreak())
    elements.append(Paragraph("Итоговая статистика", heading_style))
    
    total_errors = sum(len(r.get('errors', [])) for r in results)
    images_with_errors = sum(1 for r in results if r.get('errors'))
    
    summary_data = [
        ['Метрика', 'Значение'],
        ['Всего изображений', str(len(results))],
        ['Изображений с ошибками', str(images_with_errors)],
        ['Всего ошибок', str(total_errors)],
        ['Среднее ошибок на изображение', 
         f"{(total_errors / len(results) if results else 0):.2f}"],
        ['Лучшая архитектура (FPS)', 'RTMPose-L (65 FPS)'],
        ['Лучшая архитектура (точность)', 'HRNet-W48 (mAP 0.765)'],
    ]
    
    summary_table = Table(summary_data, colWidths=[7*cm, 6*cm])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), MAIN_FONT),
        ('FONTNAME', (0, 0), (-1, 0), BOLD_FONT),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(summary_table)
    
    doc.build(elements, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"\nPDF сохранён: {output_path} ({doc.page} стр.)")




def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj




def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    
    image_dir = 'images'
    image_files = (glob.glob(os.path.join(image_dir, '*.jpg')) + 
                   glob.glob(os.path.join(image_dir, '*.png')))
    
    if not image_files:
        print(f"В папке {image_dir} нет изображений!")
        return
    
    print(f"\nНайдено изображений: {len(image_files)}")
    
    benchmarks = benchmark_all(image_files[0])
    
    stats = {
        'total_images': len(image_files),
        'timestamp': datetime.now().isoformat(),
        'benchmarks': benchmarks,
        'architectures_compared': 5,
        'exercise': 'squat',
        'note': 'Анализ техники приседаний',
        'results': []
    }
    
    print("\nОбработка изображений...")
    for img_path in image_files:
        print(f"  {img_path}")
        
        yolo_img, yolo_data = process_image_yolo(img_path)
        if yolo_img is not None:
            cv2.imwrite(f"results/yolo_{os.path.basename(img_path)}", yolo_img)
        
        mp_img, mp_data = process_image_mediapipe(img_path)
        if mp_img is not None:
            cv2.imwrite(f"results/mp_{os.path.basename(img_path)}", mp_img)
        
        stats['results'].append({'image': img_path, **yolo_data})
    
    stats = convert_to_serializable(stats)
    
    with open('results/session_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("JSON сохранён: results/session_stats.json")
    
    generate_pdf_report(stats)
    
    print("\nГотово!")
    print(f"Обработано: {len(image_files)} изображений")
    print(f"Результаты в папке: results/")

if __name__ == '__main__':
    main()
