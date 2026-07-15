from pathlib import Path

from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem import functionspace
import dolfinx.fem.petsc
import numpy as np
import ufl


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

subdivisions = 4

number_of_training_datasets = 10
training_points = 500
testing_points = 1000

kappa_min = 0.1
kappa_max = 10.0

output_directory = Path("thermalblock")
output_directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Boundary condition
# ---------------------------------------------------------------------------

def top_boundary(x):
    """Return points located on the top boundary."""
    return np.isclose(x[1], 1.0)


# ---------------------------------------------------------------------------
# Mesh and function spaces
# ---------------------------------------------------------------------------

domain = mesh.create_unit_square(
    MPI.COMM_WORLD,
    subdivisions,
    subdivisions,
    mesh.CellType.quadrilateral,
)

V = functionspace(domain, ("CG", 1))
V_0 = functionspace(domain, ("DG", 0))


# ---------------------------------------------------------------------------
# Dirichlet boundary condition
# ---------------------------------------------------------------------------

u_D = fem.Constant(domain, 0.0)

boundary_dofs = fem.locate_dofs_geometrical(
    V,
    top_boundary,
)

boundary_condition = fem.dirichletbc(
    u_D,
    boundary_dofs,
    V,
)


# ---------------------------------------------------------------------------
# Forward problem
# ---------------------------------------------------------------------------

spatial_coordinate = ufl.SpatialCoordinate(domain)

# Unit boundary source on the left and right sides.
g = (
    ufl.conditional(
        ufl.lt(spatial_coordinate[0], 0.00001),
        1,
        0,
    )
    + ufl.conditional(
        ufl.gt(spatial_coordinate[0], 0.99999),
        1,
        0,
    )
)

trial_function = ufl.TrialFunction(V)
test_function = ufl.TestFunction(V)

kappa = fem.Function(V_0)

bilinear_form = (
    ufl.dot(
        kappa * ufl.grad(trial_function),
        ufl.grad(test_function),
    )
    * ufl.dx
)

linear_form = g * test_function * ufl.ds

forward_problem = dolfinx.fem.petsc.LinearProblem(
    bilinear_form,
    linear_form,
    bcs=[boundary_condition],
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
)


# ---------------------------------------------------------------------------
# Bottom boundary marker used by the quantity of interest
# ---------------------------------------------------------------------------

bottom_marker = ufl.conditional(
    ufl.lt(spatial_coordinate[1], 0.00001),
    1,
    0,
)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    number_of_samples,
    X_filename,
    uX_filename,
    guX_filename,
    dataset_name,
):
    """
    Generate one dataset and save it into three CSV files.

    Only the current dataset is stored in memory. Once the CSV files are
    written, the arrays can be released before generating the next dataset.
    """

    X = []
    uX = []
    guX = []

    print(
        f"Generating {dataset_name}: "
        f"{number_of_samples} samples",
        flush=True,
    )

    for sample_index in range(number_of_samples):

        # ---------------------------------------------------------------
        # Generate the random conductivity
        # ---------------------------------------------------------------

        kappa.x.array[:] = np.random.uniform(
            kappa_min,
            kappa_max,
            size=len(kappa.x.array),
        )

        kappa.x.scatter_forward()

        # A copy is required because kappa.x.array is reused at every sample.
        X.append(kappa.x.array.copy())

        # ---------------------------------------------------------------
        # Solve the forward problem
        # ---------------------------------------------------------------

        uh = forward_problem.solve()
        uh.x.scatter_forward()

        # ---------------------------------------------------------------
        # Residual and quantity of interest
        # ---------------------------------------------------------------

        residual = (
            ufl.dot(
                kappa * ufl.grad(uh),
                ufl.grad(test_function),
            )
            * ufl.dx
            - g * test_function * ufl.ds
        )

        quantity_of_interest = (
            uh * bottom_marker * ufl.ds
        )

        output_value = fem.assemble_scalar(
            fem.form(quantity_of_interest)
        )

        uX.append([output_value])

        # ---------------------------------------------------------------
        # Derivative of the residual with respect to kappa
        # ---------------------------------------------------------------

        derivative_matrix = fem.assemble_matrix(
            fem.form(
                ufl.derivative(
                    residual,
                    kappa,
                )
            )
        )

        # MatrixCSR does not require derivative_matrix.assemble().
        derivative_matrix_dense = (
            derivative_matrix.to_dense().T
        )

        # ---------------------------------------------------------------
        # Solve the adjoint problem
        # ---------------------------------------------------------------

        adjoint_left = ufl.adjoint(
            ufl.derivative(
                residual,
                uh,
            )
        )

        adjoint_right = -ufl.derivative(
            quantity_of_interest,
            uh,
        )

        adjoint_problem = fem.petsc.LinearProblem(
            adjoint_left,
            adjoint_right,
            bcs=[boundary_condition],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )

        adjoint_solution = adjoint_problem.solve()
        adjoint_solution.x.scatter_forward()

        gradient = (
            derivative_matrix_dense
            @ adjoint_solution.x.array
        )

        guX.append(gradient.copy())

    # -------------------------------------------------------------------
    # Convert only the current dataset to NumPy arrays
    # -------------------------------------------------------------------

    X_array = np.asarray(X)
    uX_array = np.asarray(uX)
    guX_array = np.asarray(guX)

    # -------------------------------------------------------------------
    # Save the current dataset
    # -------------------------------------------------------------------

    np.savetxt(
        output_directory / X_filename,
        X_array,
        delimiter=",",
    )

    np.savetxt(
        output_directory / uX_filename,
        uX_array,
        delimiter=",",
    )

    np.savetxt(
        output_directory / guX_filename,
        guX_array,
        delimiter=",",
    )
# ---------------------------------------------------------------------------
# Generate training datasets
# ---------------------------------------------------------------------------

for dataset_index in range(number_of_training_datasets):

    generate_dataset(
        number_of_samples=training_points,
        X_filename=f"train_X_dset{dataset_index}.csv",
        uX_filename=f"train_uX_dset{dataset_index}.csv",
        guX_filename=f"train_guX_dset{dataset_index}.csv",
        dataset_name=f"training dataset {dataset_index}",
    )


# ---------------------------------------------------------------------------
# Generate independent testing set
# ---------------------------------------------------------------------------

generate_dataset(
    number_of_samples=testing_points,
    X_filename="test_X.csv",
    uX_filename="test_uX.csv",
    guX_filename="test_guX.csv",
    dataset_name="testing dataset",
)

print("All training and testing datasets generated.", flush=True)
