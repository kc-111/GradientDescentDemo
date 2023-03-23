module OptimizationDemo

# Write your package code here.
using Plots
using ProgressMeter

include("my_optimizers.jl")
include("surfaces.jl")
include("contour_visualizer.jl")

export surface_ellipse
export surface_rosenbrock
export surface_himmelblau
export surface_saddleflat
export visualize_contour
export my_AdaBelief
export my_sgdm
export my_Adam
export my_Adamax
export my_AdaBeliefmax

end
