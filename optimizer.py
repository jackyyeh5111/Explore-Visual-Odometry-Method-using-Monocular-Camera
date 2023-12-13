import g2o
import numpy as np
import pdb

import numpy
import g2o


class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(g2o.Isometry3d(pose))
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement,
                 information=np.identity(6),
                 robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(g2o.Isometry3d(measurement))  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        vertex_val = list(self.vertices().values())
        return vertex_val[id].estimate()
    
# class PoseGraph(object):
#   nodes = []
#   edges = []
#   nodes_optimized = []
#   edges_optimized = []

#   def __init__(self, verbose=False):

#     """
#     Initialise the pose graph optimizer (G2O)
#     """
#     self.solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
#     self.solver=  g2o.OptimizationAlgorithmLevenberg(self.solver)

#     self.optimizer = g2o.SparseOptimizer()
#     self.optimizer.set_verbose(verbose)
#     self.optimizer.set_algorithm(self.solver)

#   def add_vertex(self, id, pose, is_fixed=False):
    
#     # Rt (pose) matrix, absolute
#     v = g2o.VertexSE3()
#     v.set_id(id)
#     v.set_estimate(g2o.Isometry3d(pose))
#     v.set_fixed(is_fixed)

#     self.optimizer.add_vertex(v)
#     self.nodes.append(v)

#   def add_edge(self, vertices, measurement=None, information=np.eye(6), robust_kernel=None):
    
#     edge = g2o.EdgeSE3()
#     for i, vertex in enumerate(vertices):
#     # check to see if we're passing in actual vertices or just the vertex ids
    
#       if isinstance(vertex, int): 
#         vertex = self.optimizer.vertex(vertex)

#       edge.set_vertex(i, vertex)
    
#     edge.set_measurement(g2o.Isometry3d(measurement)) # relative pose transformation between frames
    
#     # information matrix/precision matrix
#     # represents uncertainty of measurement error
#     # inverse of covariance matrix
#     edge.set_information(information) 
#     if robust_kernel is not None:
#       edge.set_robust_kernel(robust_kernel)
    
#     self.optimizer.add_edge(edge)
  
#   def optimize(self, max_iter=15):
#     self.optimizer.initialize_optimization()
#     self.optimizer.optimize(max_iter)

#     self.optimizer.save("data/out.g2o")

#     self.edges_optimized = []
#     if False:
#       for edge in self.optimizer.edges():
#         self.edges_optimized = [(edge.vertices()[0].estimate().matrix(), edge.vertices()[1].estimate().matrix())for edge in self.optimizer.edges()]
#     self.nodes_optimized = [i.estimate().matrix() for i in self.optimizer.vertices().values()]
#     #self.nodes_optimized = (self.nodes_optimized)
#     self.edges_optimized = np.array(self.edges_optimized)