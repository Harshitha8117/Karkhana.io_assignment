import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.spatial.distance import euclidean
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=200):
        """
        Initializes the Mobius strip parameters and mesh grid.
        :param R: Radius of the circle centerline
        :param w: Width of the strip
        :param n: Mesh resolution
        """
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.x, self.y, self.z = self._generate_mesh()

    def _generate_mesh(self):
        """Generate 3D mesh points using the parametric equations."""
        U, V = self.U, self.V
        X = (self.R + V * np.cos(U / 2)) * np.cos(U)
        Y = (self.R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def surface_area(self):
        """
        Approximate surface area using numerical integration of the magnitude of the normal vector.
        """
        def integrand(u, v):
            # Partial derivatives
            dx_du = -np.sin(u)*(self.R + v*np.cos(u/2)) - 0.5*v*np.sin(u/2)*np.cos(u)
            dx_dv = np.cos(u) * np.cos(u/2)

            dy_du = np.cos(u)*(self.R + v*np.cos(u/2)) - 0.5*v*np.sin(u/2)*np.sin(u)
            dy_dv = np.sin(u) * np.cos(u/2)

            dz_du = 0.5*v*np.cos(u/2)
            dz_dv = np.sin(u/2)

            # Cross product of partials
            n_x = dy_du * dz_dv - dz_du * dy_dv
            n_y = dz_du * dx_dv - dx_du * dz_dv
            n_z = dx_du * dy_dv - dy_du * dx_dv

            return np.sqrt(n_x**2 + n_y**2 + n_z**2)

        area, _ = dblquad(integrand, -self.w/2, self.w/2, lambda v: 0, lambda v: 2*np.pi)
        return area

    def edge_length(self):
        """
        Approximate the edge length of the Möbius strip by sampling one edge.
        """
        u_vals = np.linspace(0, 2 * np.pi, self.n)
        v = self.w / 2  # boundary edge

        x = (self.R + v * np.cos(u_vals / 2)) * np.cos(u_vals)
        y = (self.R + v * np.cos(u_vals / 2)) * np.sin(u_vals)
        z = v * np.sin(u_vals / 2)

        pts = np.column_stack((x, y, z))
        length = sum(euclidean(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        length += euclidean(pts[-1], pts[0])  # close the loop
        return length

    def plot(self):
        """Visualize the Möbius strip in 3D."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, color='cyan', edgecolor='k', alpha=0.8)
        ax.set_title('Möbius Strip')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()

    def summary(self):
        """Print summary of the Möbius strip geometric properties."""
        print("🌀 Möbius Strip Properties")
        print(f"Radius (R):       {self.R}")
        print(f"Width (w):        {self.w}")
        print(f"Resolution (n):   {self.n}")
        print(f"Surface Area ≈    {self.surface_area():.4f}")
        print(f"Edge Length ≈     {self.edge_length():.4f}")

# Example usage
if __name__ == "__main__":
    strip = MobiusStrip(R=1.0, w=0.3, n=300)
    strip.summary()
    strip.plot()
