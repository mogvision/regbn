# This file is part of RegBN: Batch Normalization of Multimodal Data 
# with Regularization.
#
# RegBN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RegBN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RegBN. If not, see <https://www.gnu.org/licenses/>.


from ARGS import args

regbn_kwargs  = {
            'gpu': args.gpu,
            'f_num_channels': 1, 
            'g_num_channels': 3,
            'f_layer_dim': [28,28],
            'g_layer_dim':[28,28],
            'normalize_input': True,
            'normalize_output': True,
            'affine': True, 
            'sigma_THR': .0, 
            'sigma_MIN': 1e-8, 
            'verbose': True,
}
