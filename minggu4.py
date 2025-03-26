import os
import numpy as np
import pandas as pd
from collections import Counter
from math import sqrt
import sys

def ensure_output_dir():
    """Ensure output directory for week 4 exists."""
    output_dir = 'output/minggu_ke_4'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    # Redirect stdout to a file
    output_dir = ensure_output_dir()
    output_file = os.path.join(output_dir, 'data_processing_output.txt')
    
    # Capture original stdout
    original_stdout = sys.stdout
    
    # Open file and redirect stdout
    with open(output_file, 'w') as f:
        sys.stdout = f
        
        try:
            print("="*50)
            print("EKSPERIMEN 1: HANDLING MISSING VALUES")
            print("="*50)
            
            # Membuat dataframe dengan missing values
            data = {'Age': [25, 30, np.nan, 35, 40, np.nan],
                    'Salary': [50000, np.nan, 60000, np.nan, 70000, 80000]}
            df = pd.DataFrame(data)
            
            print("\nData Asli:")
            print(df)
            
            # Manual Mean Imputation
            mean_age = df['Age'].mean()
            df['Age_mean'] = df['Age'].fillna(mean_age)
            
            # Manual Median Imputation
            median_age = df['Age'].median()
            df['Age_median'] = df['Age'].fillna(median_age)
            
            # Manual Most Frequent Imputation
            mode_salary = df['Salary'].mode()[0]
            df['Salary_freq'] = df['Salary'].fillna(mode_salary)
            
            print("\nSetelah Imputasi Manual:")
            print(df)
            
            print("\n" + "="*50)
            print("EKSPERIMEN 2: FEATURE SCALING MANUAL")
            print("="*50)
            
            # Data untuk scaling
            data = {'Age': [25, 30, 35, 40, 45, 50],
                    'Salary': [50000, 60000, 70000, 80000, 90000, 100000]}
            df = pd.DataFrame(data)
            
            print("\nData Asli:")
            print(df)
            
            # Manual Standardization (Z-score Normalization)
            df['Age_std'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
            df['Salary_std'] = (df['Salary'] - df['Salary'].mean()) / df['Salary'].std()
            
            # Manual Min-Max Normalization
            df['Age_norm'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
            df['Salary_norm'] = (df['Salary'] - df['Salary'].min()) / (df['Salary'].max() - df['Salary'].min())
            
            print("\nSetelah Scaling Manual:")
            print(df.round(4))
            
            print("\n" + "="*50)
            print("EKSPERIMEN 3: BINARY CLASSIFICATION EVALUATION MANUAL")
            print("="*50)
            
            # Membuat dataset klasifikasi sederhana
            # Actual values
            y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
            # Predicted values
            y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
            
            # Menghitung confusion matrix manual
            def manual_confusion_matrix(y_true, y_pred):
                TP = FP = TN = FN = 0
                for true, pred in zip(y_true, y_pred):
                    if true == 1 and pred == 1:
                        TP += 1
                    elif true == 0 and pred == 1:
                        FP += 1
                    elif true == 0 and pred == 0:
                        TN += 1
                    elif true == 1 and pred == 0:
                        FN += 1
                return np.array([[TN, FP], [FN, TP]])
            
            conf_matrix = manual_confusion_matrix(y_true, y_pred)
            print("\nConfusion Matrix Manual:")
            print(conf_matrix)
            
            # Menghitung metrik evaluasi manual
            def calculate_metrics(conf_matrix):
                TN, FP, FN, TP = conf_matrix.ravel()
                
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
                
                return accuracy, precision, recall, f1_score
            
            accuracy, precision, recall, f1_score = calculate_metrics(conf_matrix)
            
            print("\nEvaluasi Manual:")
            print(f"Akurasi: {accuracy:.4f}")
            print(f"Presisi: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1_score:.4f}")
            
            print("\n" + "="*50)
            print("EKSPERIMEN 4: HANDLING IMBALANCED DATA MANUAL")
            print("="*50)
            
            # Membuat dataset tidak seimbang
            np.random.seed(42)
            X = np.random.rand(100, 2)  # 100 sampel, 2 fitur
            y = np.array([0]*90 + [1]*10)  # 90 kelas 0, 10 kelas 1
            
            print("\nDistribusi Kelas Asli:")
            print(Counter(y))
            
            # Manual Undersampling
            def manual_undersample(X, y, target_count):
                class_0_indices = np.where(y == 0)[0]
                class_1_indices = np.where(y == 1)[0]
                
                # Kurangi kelas mayoritas
                selected_indices = np.random.choice(class_0_indices, target_count, replace=False)
                
                # Gabungkan dengan kelas minoritas
                undersampled_indices = np.concatenate([selected_indices, class_1_indices])
                
                return X[undersampled_indices], y[undersampled_indices]
            
            X_under, y_under = manual_undersample(X, y, target_count=10)
            print("\nSetelah Undersampling Manual:")
            print(Counter(y_under))
            
            # Manual Oversampling (dengan duplikasi)
            def manual_oversample(X, y, target_count):
                class_0_indices = np.where(y == 0)[0]
                class_1_indices = np.where(y == 1)[0]
                
                # Hitung berapa kali perlu duplikasi
                repeat_times = target_count // len(class_1_indices)
                remainder = target_count % len(class_1_indices)
                
                # Duplikasi sampel kelas minoritas
                oversampled_indices = np.concatenate([
                    np.repeat(class_1_indices, repeat_times),
                    np.random.choice(class_1_indices, remainder, replace=False)
                ])
                
                # Gabungkan dengan kelas mayoritas
                oversampled_indices = np.concatenate([class_0_indices, oversampled_indices])
                
                return X[oversampled_indices], y[oversampled_indices]
            
            X_over, y_over = manual_oversample(X, y, target_count=90)
            print("\nSetelah Oversampling Manual (Dengan Duplikasi):")
            print(Counter(y_over))
            
            print("\n" + "="*50)
            print("EKSPERIMEN 5: CROSS VALIDATION MANUAL")
            print("="*50)
            
            # Dataset sederhana
            X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
                          [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
            y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
            
            # Implementasi manual k-fold cross validation
            def manual_kfold_cv(X, y, k=5, random_state=None):
                if random_state is not None:
                    np.random.seed(random_state)
                    
                indices = np.arange(len(X))
                np.random.shuffle(indices)
                
                fold_sizes = np.full(k, len(X) // k, dtype=int)
                fold_sizes[:len(X) % k] += 1
                
                current = 0
                folds = []
                for fold_size in fold_sizes:
                    start, stop = current, current + fold_size
                    test_indices = indices[start:stop]
                    train_indices = np.concatenate([indices[:start], indices[stop:]])
                    folds.append((train_indices, test_indices))
                    current = stop
                
                return folds
            
            # Logistic Regression manual sederhana untuk demonstrasi
            def simple_logistic_regression(X_train, y_train, X_test, lr=0.01, epochs=100):
                # Inisialisasi weights
                weights = np.zeros(X_train.shape[1] + 1)  # +1 untuk bias
                
                # Tambahkan kolom bias
                X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
                X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]
                
                # Gradient descent
                for _ in range(epochs):
                    z = np.dot(X_train_bias, weights)
                    predictions = 1 / (1 + np.exp(-z))
                    
                    error = predictions - y_train
                    gradient = np.dot(X_train_bias.T, error) / len(y_train)
                    weights -= lr * gradient
                
                # Prediksi
                test_z = np.dot(X_test_bias, weights)
                test_pred = (1 / (1 + np.exp(-test_z))) > 0.5
                
                return test_pred.astype(int)

            # Jalankan k-fold CV
            k = 5
            folds = manual_kfold_cv(X, y, k=k, random_state=42)
            
            accuracies = []
            for i, (train_idx, test_idx) in enumerate(folds):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                y_pred = simple_logistic_regression(X_train, y_train, X_test)
                
                # Hitung akurasi
                accuracy = np.mean(y_pred == y_test)
                accuracies.append(accuracy)
                
                print(f"\nFold {i+1}:")
                print(f"Train indices: {train_idx}")
                print(f"Test indices: {test_idx}")
                print(f"Akurasi: {accuracy:.4f}")
            
            print(f"\nRata-rata Akurasi {k}-Fold CV: {np.mean(accuracies):.4f}")
        
        finally:
            # Restore stdout
            sys.stdout = original_stdout
    
    print(f"Output telah disimpan di: {output_file}")

if __name__ == "__main__":
    main()