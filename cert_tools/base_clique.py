class BaseClique(object):
    def __init__(
        self,
        Q,
        A_list,
        b_list,
        left_slice_start=[],
        left_slice_end=[],
        right_slice_start=[],
        right_slice_end=[],
    ):
        self.Q = Q
        self.A_list = A_list
        self.b_list = b_list

        self.X_dim = Q.shape[0]

        self.left_slice_start = left_slice_start
        self.left_slice_end = left_slice_end
        self.right_slice_start = right_slice_start
        self.right_slice_end = right_slice_end
