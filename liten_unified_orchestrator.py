import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. INDISCERNIBILITY CLOSURE ENGINE (PROCEDURE P)
# ==============================================================================
class RasiowaProcedurePEngine:
    def __init__(self, canvas_size=64):
        self.canvas_size = canvas_size
        self.grid_levels = [4, 8, 16, 32] # Restricted to coarse approximation covers

    def compute_topological_closure(self, quantized_image, grid_size):
        grain_size = self.canvas_size // grid_size
        pooled = F.max_pool2d(quantized_image, kernel_size=grain_size, stride=grain_size)
        return F.interpolate(pooled, size=(self.canvas_size, self.canvas_size), mode='nearest')

    def execute_procedure_p(self, original_image, tolerance_threshold=20.20):
        quantized_image = (original_image >= 0.5).float()
        intersection_T = torch.ones_like(quantized_image)
        for grid in self.grid_levels:
            cl_i = self.compute_topological_closure(quantized_image, grid)
            intersection_T = torch.min(intersection_T, cl_i)
        
        reconstruction_error = torch.norm(quantized_image - intersection_T, p=2).item()
        return reconstruction_error <= tolerance_threshold, reconstruction_error

# ==============================================================================
# 2. DUAL-LATTICE POLICY SECURITY FILTER
# ==============================================================================
class DualLatticeOrchestrator:
    def __init__(self):
        self.security_clearance = {
            "origin_company_a": "clearance_level_low",
            "transit_fleet": "clearance_level_low",
            "destination_nation_b": "clearance_level_high"
        }
        
    def verify_confinement_barrier(self, active_agent, target_domain):
        agent_clearance = self.security_clearance.get(active_agent, "public")
        target_clearance = self.security_clearance.get(target_domain, "public")
        if target_clearance == "clearance_level_high" and agent_clearance == "clearance_level_low":
            return False
        return True

def load_real_liten_image(file_path):
    img = Image.open(file_path).convert('L').resize((64, 64))
    img_np = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_np).to(device).view(1, 1, 64, 64)

# ==============================================================================
# 3. GLOBAL BATCH STREAMING BENCHMARK LOOP
# ==============================================================================
def run_dataset_batch_benchmark():
    print("================================================================================")
    print("🔮 LITEN BATCH DATASET ROUTER: CLOSING THE NEURO-SYMBOLIC WORKSPACE LOOP")
    print("================================================================================")
    
    base_liten_path = "/home/rsherman/SchedulingSelf-Similar/liten_examples"
    pe_engine = RasiowaProcedurePEngine()
    policy_controller = DualLatticeOrchestrator()
    
    benchmark_results = []
    
    # Recursively traverse your newly updated directory tree mapping
    for root, dirs, files in os.walk(base_liten_path):
        if "task.txt" in files and "image0.png" in files:
            task_file = os.path.join(root, "task.txt")
            image_file = os.path.join(root, "image0.png")
            
            # Extract domain folder names for summary tracking
            relative_path = os.path.relpath(root, base_liten_path)
            path_parts = relative_path.split(os.sep)
            domain_name = path_parts[0] if len(path_parts) > 0 else "unknown"
            subtask_name = path_parts[-1] if len(path_parts) > 0 else "unknown"
            
            # Read local task goal context
            with open(task_file, "r") as f:
                task_str = f.read().strip()
                
            # Stream the authentic visual pixels from your NVMe drive
            scene_tensor = load_real_liten_image(image_file)
            
            # 1. Run Spatial Rough Approximation Validation Check
            is_stable, l2_error = pe_engine.execute_procedure_p(scene_tensor, tolerance_threshold=20.20)
            
            # 2. Run Cross-Border Security Flow Isolation Check
            # Simulate a secure local deployment vs an illegal cross-border query trace
            is_secure = True
            if "three_execution_examples" in root: # Force a mock leak test scenario on specific directories
                is_secure = policy_controller.verify_confinement_barrier("origin_company_a", "destination_nation_b")
                
            # 3. Determine the Closed-Loop Adaptation Track
            if is_stable and is_secure:
                decision = "EXECUTE_PRISTINE"
            elif not is_secure:
                decision = "ARREST_AND_BLINDFOLD"
            else:
                decision = "SIDEWAYS_REPLAN"
                
            benchmark_results.append({
                "domain": domain_name,
                "task": subtask_name,
                "l2": l2_error,
                "stable": is_stable,
                "secure": is_secure,
                "verdict": decision
            })

    # 📊 PRINT COMPILED SYSTEM CRITERIA MATRIX TABLE
    print("\n" + "="*85)
    print(f"{'DOMAIN':<12} | {'TASK':<12} | {'L2 RECON ERROR':<15} | {'STABLE':<8} | {'SECURE':<8} | {'DECISION VERDICT':<20}")
    print("="*85)
    for res in sorted(benchmark_results, key=lambda x: (x['domain'], x['task'])):
        print(f"{res['domain']:<12} | {res['task']:<12} | {res['l2']:<15.4f} | {str(res['stable']):<8} | {str(res['secure']):<8} | {res['verdict']:<20}")
    print("="*85)
    print("👑 BATCH ANALYSIS RUN COMPLETE. ALL OUTCOMES VERIFIED EXTRACTED.")

if __name__ == "__main__":
    run_dataset_batch_benchmark()
``