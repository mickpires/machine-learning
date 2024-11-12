using Plots

x = collect(LinRange(-10,10,1000))
y = collect(LinRange(-10,10,1000))
f = x.^2 + y.^2
contour(x,y,f)
