from pathlib import Path
import csv

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("autoencoders")
OUTPUT_DIR = Path("autoencoders")

Q8_FILE = DATA_DIR / "q8.csv"
Q16_FILE = DATA_DIR / "q16.csv"

TRAINING_SET_SIZES = [100, 500]
NUM_DATASETS_PER_SIZE = 15
NUM_TEST_POINTS = 1_000

LOWER_BOUND = -2.0
UPPER_BOUND = 2.0

DTYPE = torch.float32
RANDOM_SEED = 42


def load_matrices(filename: Path) -> torch.Tensor:
    """
    Load matrices from a CSV file with this structure:

        Matrix 1
        ...
        <blank line>
        Matrix 2
        ...

    Returns
    -------
    torch.Tensor
        Tensor with shape:

        - q8.csv  -> (8, 3, 3)
        - q16.csv -> (16, 5, 5)
    """
    matrices = []
    current_matrix = []

    with filename.open("r", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            # Blank row marks the end of a matrix.
            if not row:
                if current_matrix:
                    matrices.append(current_matrix)
                    current_matrix = []
                continue

            # Skip rows such as "Matrix 1".
            if row[0].strip().lower().startswith("matrix"):
                continue

            current_matrix.append(
                [float(value) for value in row]
            )

    # Add the final matrix if the file does not end with a blank row.
    if current_matrix:
        matrices.append(current_matrix)

    if not matrices:
        raise ValueError(f"No matrices found in {filename}")

    tensor = torch.tensor(matrices, dtype=DTYPE)

    if tensor.ndim != 3:
        raise ValueError(
            f"Expected a 3D tensor from {filename}, "
            f"but got shape {tuple(tensor.shape)}."
        )

    _, rows, columns = tensor.shape

    if rows != columns:
        raise ValueError(
            f"Matrices in {filename} must be square, "
            f"but got shape {rows}x{columns}."
        )

    return tensor


def generate_quadratic_dataset(
    num_points: int,
    coefficients: torch.Tensor,
    lower_bound: float = -2.0,
    upper_bound: float = 2.0,
) -> torch.Tensor:
    """
    Generate a dataset from several quadratic forms.

    For each input vector x and each coefficient matrix Q_i:

        y_i = x^T Q_i x

    Parameters
    ----------
    num_points:
        Number of samples to generate.

    coefficients:
        Tensor of shape:

            (output_dim, input_dim, input_dim)

    lower_bound:
        Minimum value used to sample the input coordinates.

    upper_bound:
        Maximum value used to sample the input coordinates.

    Returns
    -------
    torch.Tensor
        Tensor of shape:

            (num_points, output_dim)
    """
    if coefficients.ndim != 3:
        raise ValueError(
            "coefficients must have shape "
            "(output_dim, input_dim, input_dim)."
        )

    output_dim, rows, columns = coefficients.shape

    if rows != columns:
        raise ValueError(
            "Each coefficient matrix must be square."
        )

    inputs = torch.empty(
        num_points,
        rows,
        dtype=coefficients.dtype,
        device=coefficients.device,
    ).uniform_(lower_bound, upper_bound)

    outputs = torch.einsum(
        "ni,qij,nj->nq",
        inputs,
        coefficients,
        inputs,
    )

    expected_shape = (num_points, output_dim)

    if outputs.shape != expected_shape:
        raise RuntimeError(
            f"Expected output shape {expected_shape}, "
            f"but got {tuple(outputs.shape)}."
        )

    return outputs


def save_tensor_csv(
    tensor: torch.Tensor,
    filename: Path,
) -> None:
    """Save a PyTorch tensor to a CSV file."""
    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    array = tensor.detach().cpu().numpy()

    np.savetxt(
        filename,
        array,
        delimiter=",",
        fmt="%.10g",
    )


def generate_all_datasets(
    coefficients: torch.Tensor,
    dataset_name: str,
) -> None:
    """
    Generate training and test datasets for one coefficient tensor.

    Parameters
    ----------
    coefficients:
        Tensor with shape:

            (output_dim, input_dim, input_dim)

    dataset_name:
        Subfolder name, for example "q8" or "q16".
    """
    output_dim, input_dim, _ = coefficients.shape


    # Generate training datasets.
    for num_points in TRAINING_SET_SIZES:
        ds_name = dataset_name+f"_{num_points}"
        dataset_dir = OUTPUT_DIR / ds_name
        dataset_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        for dataset_index in range(NUM_DATASETS_PER_SIZE):
            dataset = generate_quadratic_dataset(
                num_points=num_points,
                coefficients=coefficients,
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
            )

            output_file = (
                dataset_dir
                / (
                    f"train_dset{dataset_index}.csv"
                )
            )

            save_tensor_csv(
                dataset,
                output_file,
            )

            print(f"Saved {output_file}")

        # Generate test dataset.
        test_dataset = generate_quadratic_dataset(
            num_points=NUM_TEST_POINTS,
            coefficients=coefficients,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
        )
    
        test_file = (
            dataset_dir
            / "test.csv"
        )
    
        save_tensor_csv(
            test_dataset,
            test_file,
        )

        print(f"Saved {test_file}")


def main() -> None:
    torch.manual_seed(RANDOM_SEED)

    if not Q8_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {Q8_FILE.resolve()}"
        )

    if not Q16_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {Q16_FILE.resolve()}"
        )

    q8 = load_matrices(Q8_FILE)
    q16 = load_matrices(Q16_FILE)

    print(f"Loaded Q8 with shape {tuple(q8.shape)}")
    print(f"Loaded Q16 with shape {tuple(q16.shape)}")

    generate_all_datasets(
        coefficients=q8,
        dataset_name="q8",
    )

    generate_all_datasets(
        coefficients=q16,
        dataset_name="q16",
    )


if __name__ == "__main__":
    main()
