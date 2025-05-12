import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd


class irregular_convolution(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_length=9,
        batch_size=150,
        num_node=125,
        bias=True,
    ):
        """
        :param in_channel: int, The No. of channel for input matrix
        :param out_channel: int, The No. of channel for output matrix
        :param kernel_length: int, The size of irregular convolution kernel size (default: 9)
        :param batch_size: int, The size of batch (default: 150)
        :param num_node: int, The number of nodes (cells involved in this study)
        :param bias: bool, add bias in linear layer or not
        This is the core for irregular convolution: covert the convolution operation in the discrete domain
        to linear layers(weighted addition).
        """
        super(irregular_convolution, self).__init__()
        self.batch_size = batch_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_node = num_node
        self.kernel_length = kernel_length
        self.kernel_fun_bias = nn.Linear(
            self.in_channel * self.kernel_length, self.out_channel
        ).cuda()
        self.kernel_fun = nn.Linear(
            self.in_channel * self.kernel_length, self.out_channel, bias=False
        ).cuda()
        self.bias = bias

    def forward(self, x_input):
        # Detect if we're using hexagonal grid (3D input) or square grid (4D input)
        x_size = x_input.size()
        reshape_size = self.in_channel * self.kernel_length

        x_input = x_input.reshape([x_size[0], x_size[2], reshape_size])

        if self.bias:
            output = self.kernel_fun_bias(x_input).permute(0, 2, 1).unsqueeze(-1)
        else:
            output = self.kernel_fun(x_input).permute(0, 2, 1).unsqueeze(-1)
        return output


class Extraction_spatial_features(nn.Module):
    def __init__(self, kernel_size=9, batch_size=150, seq_len=24, total_nodes=125):
        """
        This module is to capture spatial dependency of bicycle usage for each interval in a sequence using irregular
        convolution operations.

        The built-in function, self.prepare_data(), is to select the semantic neighbors for each central cell.

        The function, reconstruction_file(), is to read the look-up table for mapping relationship between central cells
        and their corresponding semantic neighbors. The return of this function is the input for torch.masked_select().
        :param kernel_size: int, The size of irregular convolution kernel size (default: 9)
        :param batch_size: int, The size of batch (default: 150)
        :param seq_len: int, The length of sequence (default:24)
        :param total_nodes:The number of nodes (cells involved in this study)(default: 125 cells in New York)
        """
        super(Extraction_spatial_features, self).__init__()
        # Load the mask indicating semantic neighbors
        # self.mask should be a boolean tensor of shape (total_nodes, total_nodes)
        # where self.mask[i, j] is True if node j is a semantic neighbor of node i.
        # reconstruction_file should ensure that each node i has exactly kernel_size neighbors.
        mask_np = reconstruction_file(kernel_size=kernel_size, num_nodes=total_nodes)
        self.mask = torch.tensor(mask_np, dtype=torch.bool).cuda()

        self.batch_size = (
            batch_size  # Not strictly used in forward, batch size is dynamic
        )
        self.seq_len = seq_len
        self.total_nodes = total_nodes
        self.kernel_size = kernel_size
        self.relu = nn.ReLU(inplace=True)
        self.first_no_layer = 32
        self.second_no_layer = 16

        # Irregular convolution layers
        # Input to irregular_convolution is (batch_size, in_channels, num_nodes, kernel_size)
        # Output is (batch_size, out_channels, num_nodes, 1)
        self.irregular_layer1 = irregular_convolution(
            1,  # in_channels = 1 for the initial layer
            self.first_no_layer,
            kernel_length=self.kernel_size,
            batch_size=self.batch_size,  # batch_size param in irconv is not used
            num_node=self.total_nodes,  # num_node param in irconv is not used
        )
        self.irregular_layer2 = irregular_convolution(
            self.first_no_layer,  # in_channels = out_channels of previous layer
            self.second_no_layer,
            kernel_length=self.kernel_size,
            batch_size=self.batch_size,
            num_node=self.total_nodes,
        )
        self.batchnormal = nn.BatchNorm2d(
            self.second_no_layer
        )  # Operates on (N, C, H, W) or (N,C,L)
        # Here, it will receive (B, C2, N, 1) from irconv output
        # then permuted to (B, C2, 1, N) for BatchNorm if needed, or applied directly if BNorm handles (B,C,N,1)
        # BatchNorm2d expects input (N, C, H, W)
        # Output of irconv is (B, C_out, N, 1). This matches (N, C, H, W) if N=B, C=C_out, H=N_nodes, W=1.

        self.reduce_dimension = irregular_convolution(
            self.second_no_layer,  # in_channels = out_channels of previous layer
            1,  # out_channels = 1 to reduce feature dimension
            kernel_length=self.kernel_size,
            batch_size=self.batch_size,
            num_node=self.total_nodes,
        )

    def prepare_data(self, input_x):
        """
        This function prepares the input tensor for an irregular convolution layer.
        It gathers features from semantic neighbors for each node.
        :param input_x: Tensor of shape (batch_size, in_channels, num_nodes)
        :return: Tensor of shape (batch_size, in_channels, num_nodes, kernel_size)
        """
        B, C, N = input_x.shape
        K = self.kernel_size

        # Expand input_x to select neighbors. input_x_expanded shape: (B, C, N_nodes_to_gather_for, N_potential_neighbors)
        # input_x_expanded[b, c, i, j] will be input_x[b, c, j]
        input_x_expanded_for_gathering = input_x.unsqueeze(2).expand(B, C, N, N)

        # self.mask is (N, N). Expand it for batch and channel.
        broadcastable_mask = self.mask.unsqueeze(0).unsqueeze(0).expand(B, C, N, N)

        # Select elements using the mask. selected_elements will be a flat tensor.
        # Contains (B * C * N * K) elements.
        selected_elements = torch.masked_select(
            input_x_expanded_for_gathering, broadcastable_mask
        )

        # Reshape to (batch_size, in_channels, num_nodes, kernel_size)
        output_tensor = selected_elements.reshape(B, C, N, K)
        return output_tensor

    def forward(self, input_sequence):
        """
        This forward function captures spatial dependency of bicycle usage for each interval in a sequence.
        :param input_sequence: The sequence of historical bicycle usage.
                               Shape: (batch_size, seq_len, num_nodes, features_per_node)
                               Typically features_per_node is 1.
        :return: A tensor with captured spatial dependency, which will be input into the LSTM module.
                 Shape: (batch_size, seq_len, num_nodes)
        """
        batch_size, seq_len, num_nodes, _ = input_sequence.shape
        output_sequence = torch.empty(
            batch_size,
            self.seq_len,
            self.total_nodes,
            device=input_sequence.device,
            dtype=input_sequence.dtype,
        )

        for t_step in range(self.seq_len):
            # current_timestep_data shape: (batch_size, num_nodes, features_per_node)
            current_timestep_data = input_sequence[:, t_step, :, :]

            # Permute to (batch_size, features_per_node (in_channels), num_nodes)
            # For the first layer, features_per_node is 1, so this is (B, 1, N)
            x_for_prepare = current_timestep_data.permute(0, 2, 1)

            # Prepare data for the first irregular convolution layer
            # prepared_input1 shape: (B, 1, N, K)
            prepared_input1 = self.prepare_data(x_for_prepare)
            # cnn_output1 shape: (B, first_no_layer, N, 1)
            cnn_output1 = self.relu(self.irregular_layer1(prepared_input1))

            # Prepare data for the second irregular convolution layer
            # Input to prepare_data: (B, first_no_layer, N)
            # prepared_input2 shape: (B, first_no_layer, N, K)
            prepared_input2 = self.prepare_data(cnn_output1.squeeze(-1))
            # cnn_output2_before_bn shape: (B, second_no_layer, N, 1)
            cnn_output2_before_bn = self.irregular_layer2(prepared_input2)
            # Apply BatchNorm: BatchNorm2d expects (N,C,H,W). Here N=batch_size, C=second_no_layer, H=num_nodes, W=1
            cnn_output2 = self.relu(self.batchnormal(cnn_output2_before_bn))

            # Prepare data for the dimension reduction layer
            # Input to prepare_data: (B, second_no_layer, N)
            # prepared_input3 shape: (B, second_no_layer, N, K)
            prepared_input3 = self.prepare_data(cnn_output2.squeeze(-1))
            # reduced_output shape: (B, 1, N, 1)
            reduced_output = self.reduce_dimension(prepared_input3)
            # final_features_for_timestep shape: (B, N)
            final_features_for_timestep = (
                self.relu(reduced_output).squeeze(-1).squeeze(1)
            )

            output_sequence[:, t_step, :] = final_features_for_timestep

        return output_sequence


class Convolution_LSTM(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        num_layer=2,
        batch_size=50,
    ):
        """
        This is a LSTM module to capture the temporal dependency of bicycle usage from a historical sequence.
        :param input_channels: int, The No. of features in the input tensor.
        :param hidden_channels: int, The No. of features in the hidden state.
        :param num_layer: int, Number of recurrent layers.
        :param batch_size: int, Number of batch size.
        """
        super(Convolution_LSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layer = num_layer
        self.batch_size = batch_size
        self.lstm = nn.LSTM(
            input_size=self.input_channels,
            hidden_size=self.hidden_channels,
            num_layers=self.num_layer,
            batch_first=True,
        )
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.hidden2out_1 = nn.Linear(self.hidden_channels, 64)
        self.hidden2out_2 = nn.Linear(64, self.input_channels)

    def initalize_parameters(self, batch_size):
        return (
            Variable(
                torch.zeros(self.num_layer, batch_size, self.hidden_channels).cuda(),
                requires_grad=True,
            ),
            Variable(
                torch.zeros(self.num_layer, batch_size, self.hidden_channels).cuda(),
                requires_grad=True,
            ),
        )

    def forward(self, input):
        h0, c0 = self.initalize_parameters(input.shape[0])
        outputs, (ht, ct) = self.lstm(input, (h0, c0))
        output = ht[-1]
        output = self.hidden2out_1(output)
        output = self.hidden2out_2(output)
        output = self.tanh(output)
        output = output.unsqueeze(1).unsqueeze(-1)
        return output


class Irregular_Convolution_LSTM(nn.Module):
    def __init__(
        self,
        input_size_closeness,
        input_size_period,
        input_size_trend,
        kernel_size,
        bsize=50,
        num_node=125,
    ):
        """
        This class assembles the module capturing spatial dependency by irregular convolution and the module capturing
        temporal dependency by LSTM module. As mentioned in the paper, there are three historical periods to capture
        spatial-temporal features, respectively. The outputs of the three periods are fused by weighted addition.
        Please refer to the pre-print paper in Arxiv: https://arxiv.org/abs/2202.04376
        :param input_size_closeness: int, The size of Closeness.
        :param input_size_period: int, The size of Period.
        :param input_size_trend: int, The size of Trend.
        :param kernel_size: int, The size of convolution kernel.
        :param bsize: The size of batch.
        """
        super(Irregular_Convolution_LSTM, self).__init__()
        self.kernel_size = kernel_size
        self.bsize = bsize
        self.input_size_closeness = input_size_closeness
        self.input_size_period = input_size_period
        self.input_size_trend = input_size_trend
        self.num_node = num_node

        self.conv_module_closeness = Extraction_spatial_features(
            kernel_size=self.kernel_size,
            batch_size=self.bsize,
            seq_len=self.input_size_closeness,
            total_nodes=self.num_node,
        ).cuda()
        self.convlstm_closeness = Convolution_LSTM(
            input_channels=self.num_node,
            hidden_channels=self.num_node,
            num_layer=2,
            batch_size=self.bsize,
        ).cuda()

        self.conv_module_period = Extraction_spatial_features(
            kernel_size=self.kernel_size,
            batch_size=self.bsize,
            seq_len=self.input_size_period,
            total_nodes=self.num_node,
        ).cuda()
        self.convlstm_period = Convolution_LSTM(
            input_channels=self.num_node,
            hidden_channels=self.num_node,
            num_layer=2,
            batch_size=self.bsize,
        ).cuda()

        self.conv_module_trend = Extraction_spatial_features(
            kernel_size=self.kernel_size,
            batch_size=self.bsize,
            seq_len=self.input_size_trend,
            total_nodes=self.num_node,
        ).cuda()
        self.convlstm_trend = Convolution_LSTM(
            input_channels=self.num_node,
            hidden_channels=self.num_node,
            num_layer=2,
            batch_size=self.bsize,
        ).cuda()

        self.tanh = nn.Tanh()
        self.W_closeness, self.W_period, self.W_trend = self.init_hidden(
            [self.num_node, 1]
        )
        self.W_closeness = torch.nn.init.xavier_normal_(self.W_closeness)
        self.W_period = torch.nn.init.xavier_normal_(self.W_period)
        self.W_trend = torch.nn.init.xavier_normal_(self.W_trend)

    def forward(self, x_closeness, x_period, x_trend):
        output_closeness = self.conv_module_closeness(x_closeness)
        output_closeness = self.convlstm_closeness(output_closeness)

        output_period = self.conv_module_period(x_period)
        output_period = self.convlstm_period(output_period)

        output_trend = self.conv_module_trend(x_trend)
        output_trend = self.convlstm_trend(output_trend)

        output = (
            self.W_closeness * output_closeness
            + self.W_period * output_period
            + self.W_trend * output_trend
        )
        output = self.tanh(output)
        return output

    def init_hidden(self, shape):
        return (
            torch.nn.Parameter(torch.empty(shape[0], shape[1]), requires_grad=True),
            torch.nn.Parameter(torch.empty(shape[0], shape[1]), requires_grad=True),
            torch.nn.Parameter(torch.empty(shape[0], shape[1]), requires_grad=True),
        )


def reconstruction_file(kernel_size, num_nodes):  # Added num_nodes parameter
    """
    Reads a full similarity matrix CSV of shape (N x N) with header and index,
    then for each node picks its top-k most similar *other* nodes and returns
    a binary mask matrix of shape (N, N) where mask[i, j] = 1 iff j is among
    the top-k neighbors of i.

    :param kernel_size: number of neighbors to select per node.
    :param num_nodes: total number of nodes, to verify consistency.
    :return: mask: np.ndarray of shape (N, N), dtype=int.
    """
    # 1) Load the full similarity dataframe
    similarity_csv_path = "/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/IrConv-LSTM/data/similarity_matrix.csv"
    df = pd.read_csv(similarity_csv_path, index_col=0)

    # 2) Build a mapping from node-id to row index and verify num_nodes
    node_ids_from_csv = df.index.tolist()
    N_csv = len(node_ids_from_csv)
    if N_csv != num_nodes:
        raise ValueError(
            f"Number of nodes in similarity_matrix.csv ({N_csv}) does not match model's num_nodes ({num_nodes})."
        )

    # Ensure columns match index for a symmetric similarity matrix
    if not all(df.columns == df.index):
        raise ValueError("Similarity matrix CSV columns do not match index.")

    node_to_idx = {node: i for i, node in enumerate(node_ids_from_csv)}

    # 3) Initialize the binary mask
    mask = np.zeros((N_csv, N_csv), dtype=int)

    # 4) For each node, pick its top-k neighbors
    for i, node_i_label in enumerate(node_ids_from_csv):
        sim_row = df.loc[
            node_i_label
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning
        # Exclude self by dropping it (or setting its similarity to a very low value if not present)
        if node_i_label in sim_row.index:
            sim_row = sim_row.drop(node_i_label)
        else:
            # This case should ideally not happen if the matrix is N x N and includes self-similarity
            pass

        # Pick the labels of the top-k highest similarities
        # Ensure we don't ask for more neighbors than available (excluding self)
        actual_kernel_size = min(kernel_size, len(sim_row))
        if actual_kernel_size < kernel_size:
            print(
                f"Warning: Node {node_i_label} has only {actual_kernel_size} other nodes, requested {kernel_size} neighbors."
            )

        if actual_kernel_size > 0:
            topk_node_labels = sim_row.nlargest(actual_kernel_size).index.tolist()
            for node_j_label in topk_node_labels:
                j = node_to_idx[node_j_label]
                mask[i, j] = 1
        # If actual_kernel_size is 0 (e.g. only one node in total), no neighbors are set.
    return mask
