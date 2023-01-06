import numpy as np

class LoadMuldiconProperties:
    def __init__(self,data_directory):
        self.geom = self._Geom(data_directory)
        self.flightconditions = self._FlightConditions()

    class _FlightConditions:
        def __init__(self):
            # Aircraft properties and flight conditions #
            self.rho = 1.225 #kg/m^3
            self.V = 68.06 #m/s
            self.ref_S = 77.876 #m^2
            self.ref_L = 6.0 #m

    class _Geom:
        def __init__(self,data_directory):
            """

            Load geometry properties of the UCAV MULDICON.

            """
            try:
                # Load original geometry file #
                self.geom_coordinates = np.load(data_directory + 'POD/geom_original.npy')  # geometry coordinates (X,Y,Z)

                # Split coordinates in upper and lower surface #
                self.geom_x = self.geom_coordinates[:, :, 0]  # lower surface x-coordinates
                self.geom_x_lower = self.geom_coordinates[0:int(len(self.geom_coordinates) / 2) + 1, :, 0]  # lower surface x-coordinates
                self.geom_x_upper = self.geom_coordinates[int(len(self.geom_coordinates) / 2):len(self.geom_coordinates), :, 0]  # upper surface x-coordinates
                
                self.geom_y = self.geom_coordinates[:, :, 1]
                self.geom_y_lower = self.geom_coordinates[0:int(len(self.geom_coordinates) / 2) + 1, :, 1]  # lower surface y-coordinates
                self.geom_y_upper = self.geom_coordinates[int(len(self.geom_coordinates) / 2):len(self.geom_coordinates), :, 1]  # upper surface y-coordinates

                # Convert coordinates to normalized coordinates #
                self.geom_x_norm = self.geom_x / max(self.geom_x[:, -1])
                self.geom_x_lower_norm = self.geom_x_lower / max(self.geom_x_lower[:, -1])
                self.geom_x_upper_norm = self.geom_x_upper / max(self.geom_x_upper[:, -1])
                
                self.geom_y_norm = self.geom_y / max(self.geom_y[:, -1])
                self.geom_y_lower_norm = self.geom_y_lower / max(self.geom_y_lower[:, -1])
                self.geom_y_upper_norm = self.geom_y_upper / max(self.geom_y_upper[:, -1])
            except:
                print("MULDICON geometry files not found. Application will close.")
                exit()

            try:
                # Load vertex coordinates, normal vectors and cell area #
                self.geom_normal_coordinates = np.load(data_directory + 'POD/geom_original_cell_coordinates.npy')
                self.geom_normal_distance_reference = np.load(
                    data_directory + 'POD/geom_original_cell_reference_distances.npy')
                self.geom_normal_vectors = np.load(data_directory + 'POD/geom_original_normal_vectors.npy')
                self.geom_normal_area = np.load(data_directory + 'POD/geom_original_cell_area.npy')
            except:
                print("MULDICON vertex files not found. Application will close.")
                exit()
