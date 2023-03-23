# Elliptic Bowl
surface_ellipse = (x, y) -> 17/80*x^2+1/10*y^2-12/80*x*y 

# rosenbrock, the constant in front of (y-x^2)^2 controls valley flatness
surface_rosenbrock = (x, y) -> (x-1)^2 + 10*(y-x^2)^2 

# Himmelblau's function
surface_himmelblau = (x, y) -> (x^2+y-11)^2+(x+y^2-7)^2

# Saddle flat from https://jermwatt.github.io/machine_learning_refined/notes/3_First_order_methods/3_9_Normalized.html
surface_saddleflat = (x, y) -> tanh(4*x+4*y)+maximum([0.4*y^2,1 ]) + 1
