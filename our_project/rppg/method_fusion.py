import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class MultiMethodFusion:
    """
    Machine Learning model designed to fuse the outputs of multiple algorithms 
    (Classical and Deep Learning) into a single, high-fidelity HR estimate. (Phase 26)
    """
    def __init__(self, model_type="xgboost", model_path=None):
        self.model_type = model_type
        self.model_path = model_path
        
        if model_type == "rf":
            self.model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == "xgboost":
            self.model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        else:
            raise ValueError("Unsupported model type. Choose 'rf' or 'xgboost'.")
            
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def _prepare_features(self, method_outputs, motion_score, illum_score):
        """
        Converts the raw outputs into a feature vector for the ML model.
        
        method_outputs is a dict:
        {
            "chrom": {"hr": 72.1, "sqi": 0.8},
            "pos": {"hr": 71.5, "sqi": 0.75},
            "green": {"hr": 73.0, "sqi": 0.4},
            "physnet": {"hr": 72.0, "sqi": 0.9}
        }
        """
        # Ensure consistent order
        methods = ["chrom", "pos", "green", "physnet"]
        features = []
        
        for m in methods:
            if m in method_outputs:
                features.extend([method_outputs[m]["hr"], method_outputs[m]["sqi"]])
            else:
                features.extend([0.0, 0.0]) # Missing data padding
                
        # Append environmental/contextual features
        features.extend([motion_score, illum_score])
        
        return np.array(features)

    def train(self, X_train, y_train):
        """
        Trains the fusion model on historical method outputs vs Ground Truth.
        X_train shape: (N_samples, n_features)
        y_train shape: (N_samples,)
        """
        print(f"Training {self.model_type.upper()} Fusion Model on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
    def predict(self, method_outputs, motion_score, illum_score):
        """
        Predicts the true HR given the outputs of all active methods.
        """
        if not self.is_trained:
            # Fallback: Just return the HR of the method with the highest SQI
            best_hr = 0.0
            best_sqi = -1.0
            for m, data in method_outputs.items():
                if data["sqi"] > best_sqi:
                    best_sqi = data["sqi"]
                    best_hr = data["hr"]
            return best_hr
            
        features = self._prepare_features(method_outputs, motion_score, illum_score)
        features = features.reshape(1, -1)
        
        prediction = self.model.predict(features)[0]
        return float(prediction)

    def save_model(self, path):
        if self.is_trained:
            joblib.dump(self.model, path)
            
    def load_model(self, path):
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_trained = True

if __name__ == "__main__":
    fusion = MultiMethodFusion(model_type="xgboost")
    
    # Generate dummy training data
    # Columns: [chrom_hr, chrom_sqi, pos_hr, pos_sqi, green_hr, green_sqi, phys_hr, phys_sqi, motion, illum]
    X_dummy = np.random.rand(100, 10) * 100
    y_dummy = np.random.rand(100) * 100
    
    fusion.train(X_dummy, y_dummy)
    
    sample_methods = {
        "chrom": {"hr": 72.1, "sqi": 0.8},
        "pos": {"hr": 71.5, "sqi": 0.75},
        "green": {"hr": 73.0, "sqi": 0.4},
        "physnet": {"hr": 72.0, "sqi": 0.9}
    }
    
    fused_hr = fusion.predict(sample_methods, motion_score=0.1, illum_score=0.9)
    print(f"Fused HR Prediction: {fused_hr:.2f} BPM")
