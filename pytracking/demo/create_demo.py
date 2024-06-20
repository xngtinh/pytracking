import cv2
import numpy as np
import torch
from collections import Counter
import pandas as pd
from scipy.stats import pearsonr

# Assuming you have a trained PyTorch model and a preprocessing function
model = ...  # Load your trained model here
preprocess = ...  # Define your preprocessing function here
video_path = 'path_to_your_video.mp4'
ground_truth_path = 'path_to_ground_truth.csv'  # CSV file with ground truth data
output_csv = 'experiment_results_with_error.csv'

def extract_frame_metrics(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Brightness
    brightness = np.mean(gray_frame)
    
    # Blurriness
    blurriness = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    
    return brightness, blurriness

def process_video(video_path, model, preprocess, ground_truth):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    results = []
    
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame metrics
        brightness, blurriness = extract_frame_metrics(frame)
        
        # Preprocess frame for model input
        input_tensor = preprocess(frame)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        # Model inference
        with torch.no_grad():
            outputs = model(input_tensor)
        
        for output in outputs:
            labels = output['labels'].detach().cpu().numpy()
            scores = output['scores'].detach().cpu().numpy()
            
            # Confidence Scores
            confidence_scores = scores.tolist()
            
            # Number of Classes
            num_classes = len(Counter(labels))
            
            # Calculate Absolute Error (example, needs actual ground truth comparison)
            # This part assumes you have a way to calculate the absolute error for your application
            # Replace the following line with actual error calculation
            absolute_error = calculate_absolute_error(frame_index, ground_truth, output)  # Implement this function
            
            # Store results
            results.append({
                'frame_index': frame_index,
                'brightness': brightness,
                'blurriness': blurriness,
                'confidence_scores': confidence_scores,
                'num_classes': num_classes,
                'absolute_error': absolute_error
            })
        
        frame_index += 1
    
    cap.release()
    
    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    return df

def calculate_absolute_error(frame_index, ground_truth, output):
    # Implement this function to calculate the absolute error based on ground truth and model output
    # For example, if ground truth and predictions are bounding boxes, you might calculate the IoU or L2 distance
    # This is a placeholder implementation
    ground_truth_labels = ground_truth[frame_index]['labels']
    predicted_labels = output['labels'].detach().cpu().numpy()
    error = np.abs(len(ground_truth_labels) - len(predicted_labels))  # Example error calculation
    return error

def compute_correlation(df):
    # Calculate correlations between factors and absolute error
    correlations = {}
    factors = ['brightness', 'blurriness', 'confidence_scores', 'num_classes']
    
    for factor in factors:
        if factor == 'confidence_scores':
            # Compute the mean confidence score for correlation
            mean_confidence_scores = df[factor].apply(np.mean)
            correlation, _ = pearsonr(mean_confidence_scores, df['absolute_error'])
        else:
            correlation, _ = pearsonr(df[factor], df['absolute_error'])
        
        correlations[factor] = correlation
    
    return correlations

# Example of preprocessing function (modify according to your model's requirements)
def preprocess(frame):
    # Example: Convert frame to tensor and normalize
    frame = cv2.resize(frame, (224, 224))  # Resize if needed
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0  # Normalize to [0, 1]
    frame = np.transpose(frame, (2, 0, 1))  # Convert to CHW format
    frame = torch.tensor(frame, dtype=torch.float32)
    return frame

# Load ground truth data
ground_truth = pd.read_csv(ground_truth_path).to_dict('records')

# Run the processing function
df = process_video(video_path, model, preprocess, ground_truth)

# Compute and print correlations
correlations = compute_correlation(df)
print(correlations)

####
import cv2
import numpy as np
import torch
from scipy.stats import pearsonr

# Assuming you have a trained PyTorch model and a preprocessing function
model = ...  # Load your trained model here
preprocess = ...  # Define your preprocessing function here

def initialize_kalman_filter():
    # Initialize a Kalman filter with appropriate dimensions and noise parameters
    kf = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, vx, vy), 2 measurement variables (x, y)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
    return kf

def predict_kalman_filter(kf):
    return kf.predict()

def update_kalman_filter(kf, measurement):
    kf.correct(measurement)

def run_deep_learning_tracker(frame, model, preprocess):
    input_tensor = preprocess(frame)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        outputs = model(input_tensor)
    
    return outputs

def run_klt_tracker(prev_frame, curr_frame, prev_points):
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    curr_points, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_points, None, **lk_params)
    return curr_points, st, err

def main(video_path, model, preprocess):
    cap = cv2.VideoCapture(video_path)
    kf = initialize_kalman_filter()

    ret, prev_frame = cap.read()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    points_to_track = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if points_to_track is not None:
            # KLT tracking
            points_to_track, st, err = run_klt_tracker(prev_frame_gray, frame_gray, points_to_track)
            klt_points = points_to_track[st == 1]
        
        # Deep learning tracking
        outputs = run_deep_learning_tracker(frame, model, preprocess)
        for output in outputs:
            labels = output['labels'].detach().cpu().numpy()
            scores = output['scores'].detach().cpu().numpy()
            boxes = output['boxes'].detach().cpu().numpy()
            
            # For simplicity, use the center of the first detected box as the measurement
            if len(boxes) > 0:
                box = boxes[0]
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                measurement = np.array([[np.float32(x_center)], [np.float32(y_center)]])
                
                # Kalman filter update with deep learning measurement
                update_kalman_filter(kf, measurement)
                
                # Get Kalman filter prediction
                kalman_prediction = predict_kalman_filter(kf)
                x_pred, y_pred = kalman_prediction[0], kalman_prediction[1]
                
                # Visualize the tracking results
                cv2.circle(frame, (int(x_pred), int(y_pred)), 5, (0, 255, 0), -1)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        prev_frame_gray = frame_gray.copy()
        points_to_track = klt_points.reshape(-1, 1, 2) if points_to_track is not None else None
    
    cap.release()
    cv2.destroyAllWindows()

# Example of preprocessing function (modify according to your model's requirements)
def preprocess(frame):
    # Example: Convert frame to tensor and normalize
    frame = cv2.resize(frame, (224, 224))  # Resize if needed
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0  # Normalize to [0, 1]
    frame = np.transpose(frame, (2, 0, 1))  # Convert to CHW format
    frame = torch.tensor(frame, dtype=torch.float32)
    return frame

# Run the main function
video_path = 'path_to_your_video.mp4'
main(video_path, model, preprocess)