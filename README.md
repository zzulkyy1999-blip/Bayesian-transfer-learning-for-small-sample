import numpy as np

# =====================================================
# 1. Hyperparameter Update (Property 2 from paper)
# =====================================================
def hypar_update(n, D, a, b, b_ts, u, v):
    """
    Updates hyperparameters based on the provided dataset and priors.
    
    Args:
        n (int): Sample size.
        D (array-like): Data samples.
        a (float): Shape parameter (alpha).
        b (float): Scale parameter (beta).
        b_ts (float): Transfer scale parameter derived from correlation.
        u (float): Mean prior (mu).
        v (float): Variance scaling prior (nu).
        
    Returns:
        tuple: Updated (v_new, u_new, a_new, b_new).
    """
    x_bar = np.mean(D)
    S = np.var(D, ddof=0) # Population variance used in this context

    v_new = v + n
    u_new = (v * u + n * x_bar) / (v + n)
    a_new = a + n / 2
    
    # Update rule for b (beta)
    # Note: Ensure b_ts is not zero to avoid division by zero
    denom_term = (b / b_ts) + (n * S / 2) + (n * v * (x_bar - u) ** 2 / (2 * (n + v)))
    if denom_term == 0:
        b_new = b # Fallback if denominator is zero
    else:
        b_new = 1.0 / denom_term

    return v_new, u_new, a_new, b_new

# =====================================================
# 2. Prediction Function
# =====================================================
def bayesian_transfer_predict(
    D_t_train,
    D_s,
    rho,
    a=10,
    b_t=1,
    b_s=1,
    v_t=10,
    v_s=10
):
    """
    Performs Bayesian transfer prediction.
    """
    n_t = len(D_t_train)
    n_s = len(D_s)
    
    # Calculate empirical means
    u_t = np.mean(D_t_train)
    u_s = np.mean(D_s)
    
    # Calculate transfer term based on correlation rho
    b_ts = (1 - rho) * b_t * b_s
    
    # Update Target parameters using Source influence
    # FIX: Use u_s (Source Mean) as the Prior 'u' for the Target update.
    # This allows the source domain to influence the target prediction.
    # Previously: hypar_update(..., u_t, v_t) -> No transfer occurred.
    _, u_t_star, _, _ = hypar_update(n_t, D_t_train, a, b_s, b_ts, u_s, v_t)
    
    # Source update (calculated for completeness)
    _ , _ , _ , _  = hypar_update(n_s, D_s, a, b_t, b_ts, u_s, v_s)
    
    return u_t_star

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    return abs(y_true - y_pred) / abs(y_true) * 100

# =====================================================
# 3. Main Execution
# =====================================================
if __name__ == "__main__":
    # Remove fixed rho, it will be set dynamically based on source group
    # rho = 0.9 

    # Target Data (Fixed)
    # 目标域数据 (target_train用于训练，target_test用于验证)
    target_data = {
        "Compressive": {"train": np.array([6100, 6302, 6210]), "test": 6288},
        "Failure":     {"train": np.array([5796, 5988, 5898]), "test": 5970},
        "Yield":       {"train": np.array([4864, 5010, 4992]), "test": 4968}
    }

    # All Source Domains Data
    # 所有源域数据
    source_datasets = {
        "L1": {
            "Compressive": np.array([5176, 4982, 5008, 5082, 5044]),
            "Failure":     np.array([4654, 4480, 4500, 4564, 4532]),
            "Yield":       np.array([4369, 4232, 4138, 4320, 4174])
        },
        "L2": {
            "Compressive": np.array([7318, 7446, 7572, 7412, 7280]),
            "Failure":     np.array([6592, 6696, 6812, 6668, 6550]),
            "Yield":       np.array([5648, 5666, 5724, 5572, 5618])
        },
        "L3": {
            "Compressive": np.array([5276, 5326, 5354, 5250, 5300]),
            "Failure":     np.array([4748, 4784, 4814, 4726, 4764]),
            "Yield":       np.array([4038, 4062, 4100, 3994, 4028])
        },
        "L4": {
            "Compressive": np.array([7568, 7574, 7608, 7496, 7580]),
            "Failure":     np.array([6816, 6816, 6842, 6744, 6820]),
            "Yield":       np.array([5702, 5636, 5616, 5574, 5624])
        },
        "L5": {
            "Compressive": np.array([4788, 4726, 4814, 4816, 4750]),
            "Failure":     np.array([4544, 4486, 4570, 4572, 4510]),
            "Yield":       np.array([4520, 4478, 4532, 4554, 4500])
        },
        "M1": {
            "Compressive": np.array([6496, 6580, 6536, 6486, 6484]),
            "Failure":     np.array([6168, 6248, 6212, 6164, 6160]),
            "Yield":       np.array([5532, 5574, 5540, 5520, 5580])
        },
        "M2": {
            "Compressive": np.array([7598, 7540, 7738, 7678, 7632]),
            "Failure":     np.array([7220, 7162, 7350, 7306, 7248]),
            "Yield":       np.array([6312, 6336, 6412, 6302, 6368])
        },
        "M3": {
            "Compressive": np.array([5608, 5736, 5722, 5630, 5482]),
            "Failure":     np.array([5326, 5448, 5426, 5348, 5206]),
            "Yield":       np.array([4896, 5010, 5014, 4910, 4886])
        },
        "M4": {
            "Compressive": np.array([6510, 6650, 6630, 6532, 6426]),
            "Failure":     np.array([6184, 6318, 6296, 6204, 6102]),
            "Yield":       np.array([5360, 5520, 5508, 5390, 5332])
        },
        "M5": {
            "Compressive": np.array([6528, 6554, 6568, 6616, 6570]),
            "Failure":     np.array([6200, 6228, 6238, 6288, 6240]),
            "Yield":       np.array([5604, 5628, 5572, 5652, 5646])
        },
        "H1": {
            "Compressive": np.array([6500, 6594, 6526, 6456, 6526]),
            "Failure":     np.array([6170, 6272, 6194, 6128, 6196]),
            "Yield":       np.array([5378, 5442, 5436, 5322, 5368])
        },
        "H2": {
            "Compressive": np.array([6632, 6636, 6584, 6572, 6572]),
            "Failure":     np.array([6300, 6302, 6254, 6240, 6236]),
            "Yield":       np.array([5606, 5534, 5626, 5552, 5552])
        },
        "H3": {
            "Compressive": np.array([6568, 6558, 6516, 6564, 6410]),
            "Failure":     np.array([6236, 6230, 6184, 6238, 6088]),
            "Yield":       np.array([5358, 5350, 5244, 5372, 5266])
        },
        "H4": {
            "Compressive": np.array([6474, 6522, 6458, 6536, 6388]),
            "Failure":     np.array([6146, 6194, 6140, 6206, 6066]),
            "Yield":       np.array([5310, 5294, 5252, 5326, 5218])
        },
        "H5": {
            "Compressive": np.array([6112, 6258, 6298, 6292, 6386]),
            "Failure":     np.array([5802, 5946, 5978, 5978, 6066]),
            "Yield":       np.array([5016, 5122, 5116, 5066, 5208])
        }
    }

    # Print Header
    print(f"{'Source':<8} | {'Rho':<5} | {'Compressive MAPE':<18} | {'Failure MAPE':<14} | {'Yield MAPE':<12} | {'Avg MAPE':<10}")
    print("-" * 85)

    # Iterate over each Source Domain
    for source_name, source_data in source_datasets.items():
        # Set rho based on group (L=0.1, M=0.5, H=0.9)
        if source_name.startswith("L"):
            rho = 0.1
        elif source_name.startswith("M"):
            rho = 0.5
        elif source_name.startswith("H"):
            rho = 0.9
        else:
            rho = 0.9 # Default fallback

        mapes = []
        
        # Calculate for each property (Compressive, Failure, Yield)
        for prop in ["Compressive", "Failure", "Yield"]:
            # Get Source Data
            D_s = source_data[prop]
            
            # Get Target Data
            D_t_train = target_data[prop]["train"]
            y_true = target_data[prop]["test"]
            
            # Predict
            pred = bayesian_transfer_predict(
                D_t_train,
                D_s,
                rho=rho
            )
            
            # Calculate MAPE
            err = mape(y_true, pred)
            mapes.append(err)
        
        # Print row for this source
        print(f"{source_name:<8} | {rho:<5.1f} | {mapes[0]:<18.2f} | {mapes[1]:<14.2f} | {mapes[2]:<12.2f} | {np.mean(mapes):<10.2f}")

    print("-" * 85)
