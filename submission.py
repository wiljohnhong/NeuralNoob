import collections
import copy
import functools
import math
import os
from collections import Sequence

import nmmo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ijcai2022nmmo import Team
from ijcai2022nmmo.config import CompetitionConfig
from nmmo.io import action
from nmmo.io.action import Direction

N_ACHIEVEMENT = 4
N_SUB_ACHIEVEMENT = 3
N_TEAM = 16
N_PLAYER_PER_TEAM = 8
N_PLAYER = N_TEAM * N_PLAYER_PER_TEAM
MAX_STEP = 1024
TERRAIN_SIZE = 160
MAP_LEFT = 16
MAP_RIGHT = 144
MAP_SIZE = MAP_RIGHT - MAP_LEFT + 1
WINDOW = 15
WINDOW_CENTER = 7

MOVE_ACT_MEANINGS = {
    0: 'north',
    1: 'south',
    2: 'east',
    3: 'west',
    4: 'no-ops',
}
ATK_TYPE_MEANINGS = {
    0: 'melee',
    1: 'range',
    2: 'mage',
}
ATK_TARGET_MEANINGS = {
    0: 'enemy_1',
    1: 'enemy_2',
    2: 'npc_1',
    3: 'npc_2',
}


N_MOVE_ACTION = 5
N_ATK_TYPE = 3
N_ATK_TARGET = 4
N_ATK_ACTION = N_ATK_TYPE * N_ATK_TARGET + 1
N_ACTION = N_MOVE_ACTION * N_ATK_ACTION + 1

N_FEAT_PER_TARGET = 45
N_FEAT_PER_ALLY = 23
N_FEAT_SELF = 182

N_CH = 30

CH_HAS_ENTITY = 0
CH_TILE_LAVA = 1
CH_TILE_WATER = 2
CH_TILE_GRASS = 3
CH_TILE_SCRUB = 4
CH_TILE_FOREST = 5
CH_TILE_STONE = 6
CH_POS_X = 7
CH_POS_Y = 8
CH_SPAWN_DIST_DIFF = 9
CH_FURTHEST_EXPLORE_DIFF = 11
CH_ATTACK_AREA = 12
CH_ENTITY_OUR = 13
CH_ENTITY_ENEMY = 14
CH_ENTITY_NPC_PASSIVE = 15
CH_ENTITY_NPC_NEUTRAL = 16
CH_ENTITY_NPC_HOSTILE = 17
CH_ATTACKER_ME = 18
CH_ATTACKER_OUR = 19
CH_ATTACKER_ENEMY = 20
CH_ATTACKER_NPC = 21
CH_LEVEL = 22
CH_DAMAGE = 23
CH_FOOD = 24
CH_WATER = 25
CH_HEALTH = 26
CH_FROZEN = 27
CH_PROGRESS = 28
CH_HISTORY_PATH = 29

TO_STACK_CH = [
    CH_HAS_ENTITY, CH_ENTITY_OUR, CH_ENTITY_ENEMY,
    CH_ENTITY_NPC_PASSIVE, CH_ENTITY_NPC_NEUTRAL, CH_ENTITY_NPC_HOSTILE,
    CH_DAMAGE, CH_FOOD, CH_WATER, CH_HEALTH, CH_FROZEN,
]


def single_as_batch(func):
    def _recursive_processing(x, squeeze=False):
        if isinstance(x, Sequence):
            return (_recursive_processing(_, squeeze) for _ in x)
        elif isinstance(x, dict):
            return {k: _recursive_processing(v, squeeze) for k, v in x.items()}
        else:
            return x.squeeze(0) if squeeze else x.unsqueeze(0)

    @functools.wraps(func)
    def wrap(self, *tensors):
        tensors = _recursive_processing(tensors)
        result = func(self, *tensors)
        return _recursive_processing(result, squeeze=True)

    return wrap


def same_padding(in_size, filter_size, stride_size):
    in_height, in_width = in_size
    if isinstance(filter_size, int):
        filter_height, filter_width = filter_size, filter_size
    else:
        filter_height, filter_width = filter_size
    stride_height, stride_width = stride_size

    out_height = np.ceil(float(in_height) / float(stride_height))
    out_width = np.ceil(float(in_width) / float(stride_width))

    pad_along_height = int(
        ((out_height - 1) * stride_height + filter_height - in_height))
    pad_along_width = int(
        ((out_width - 1) * stride_width + filter_width - in_width))
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    output = (out_height, out_width)
    return padding, output


class SlimConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding,
                 initializer="default", activation_fn=None, bias_init=0):
        super(SlimConv2d, self).__init__()
        layers = []

        # Padding layer.
        if padding:
            layers.append(nn.ZeroPad2d(padding))

        # Actual Conv2D layer (including correct initialization logic).
        conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        if initializer:
            if initializer == "default":
                initializer = nn.init.xavier_uniform_
            initializer(conv.weight)
        nn.init.constant_(conv.bias, bias_init)
        layers.append(conv)
        if activation_fn is not None:
            layers.append(activation_fn())

        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


class ResidualBlock(nn.Module):
    def __init__(self, i_channel, o_channel, in_size, kernel_size=3, stride=1):
        super().__init__()
        self._relu = nn.ReLU(inplace=True)

        padding, out_size = same_padding(in_size, kernel_size, [stride, stride])
        self._conv1 = SlimConv2d(i_channel, o_channel,
                                 kernel=3, stride=stride,
                                 padding=padding, activation_fn=None)

        padding, out_size = same_padding(out_size, kernel_size, [stride, stride])
        self._conv2 = SlimConv2d(o_channel, o_channel,
                                 kernel=3, stride=stride,
                                 padding=padding, activation_fn=None)

        self.padding, self.out_size = padding, out_size

    def forward(self, x):
        out = self._relu(x)
        out = self._conv1(out)
        out = self._relu(out)
        out = self._conv2(out)
        out += x
        return out


class ResNet(nn.Module):
    def __init__(self, in_ch, in_size, channel_and_blocks=None):
        super().__init__()

        out_size = in_size
        conv_layers = []
        if channel_and_blocks is None:
            channel_and_blocks = [(16, 2), (32, 2), (32, 2)]

        for (out_ch, num_blocks) in channel_and_blocks:
            # Downscale
            padding, out_size = same_padding(out_size, filter_size=3,
                                             stride_size=[1, 1])
            conv_layers.append(
                SlimConv2d(in_ch, out_ch, kernel=3, stride=1, padding=padding,
                           activation_fn=None))

            padding, out_size = same_padding(out_size, filter_size=3,
                                             stride_size=[2, 2])
            conv_layers.append(nn.ZeroPad2d(padding))
            conv_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

            # Residual blocks
            for _ in range(num_blocks):
                res = ResidualBlock(i_channel=out_ch, o_channel=out_ch,
                                    in_size=out_size)
                conv_layers.append(res)

            padding, out_size = res.padding, res.out_size
            in_ch = out_ch

        conv_layers.append(nn.ReLU(inplace=True))
        self.resnet = nn.Sequential(*conv_layers)

    def forward(self, x):
        out = self.resnet(x)
        return out


class NMMOModel(nn.Module):
    def __init__(self):
        super().__init__()

        n_action = 66
        n_vec_in = 701

        in_ch1 = 52
        in_size1 = [15, 15]
        self.resnet1 = ResNet(in_ch1, in_size1, [[32, 2], [64, 2]])
        sample_input1 = torch.zeros(1, in_ch1, *in_size1)
        with torch.no_grad():
            self.n_hidden1 = len(self.resnet1(sample_input1).flatten())

        in_ch2 = 2
        in_size2 = [33, 33]
        self.resnet2 = ResNet(in_ch2, in_size2, [[16, 2], [32, 2], [32, 2]])
        sample_input2 = torch.zeros(1, in_ch2, *in_size2)
        with torch.no_grad():
            self.n_hidden2 = len(self.resnet2(sample_input2).flatten())

        self.img_fc = nn.Sequential(
            nn.Linear(self.n_hidden1, 512),
            nn.ReLU(),
        )
        self.big_img_fc = nn.Sequential(
            nn.Linear(self.n_hidden2, 512),
            nn.ReLU(),
        )
        self.vec_fc = nn.Sequential(
            nn.Linear(n_vec_in, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        n_comb_hidden = 512 + 512 + 512
        n_comb_out = 256
        self.comb_layer = nn.Sequential(
            nn.Linear(n_comb_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, n_comb_out),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(n_comb_out, 128),
            nn.ReLU(),
            nn.Linear(128, n_action),
        )
        self.value_head = nn.Sequential(
            nn.Linear(N_PLAYER_PER_TEAM * n_comb_out, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    @single_as_batch
    def infer(self, state):
        return self.forward(state)

    def forward(self, state):
        # state['img']: batch, agent, channel, x, y
        x_big_img, x_img, x_vec = state['big_img'], state['img'], state['vec']
        batch_size = x_img.shape[0]

        x_img = x_img.view(-1, *x_img.shape[-3:])
        img_hidden = self.resnet1(x_img).reshape(batch_size, N_PLAYER_PER_TEAM, self.n_hidden1)
        img_hidden = self.img_fc(img_hidden)
        x_big_img = x_big_img.view(-1, *x_big_img.shape[-3:])
        big_img_hidden = self.resnet2(x_big_img).reshape(batch_size, N_PLAYER_PER_TEAM, self.n_hidden2)
        big_img_hidden = self.big_img_fc(big_img_hidden)
        vec_hidden = self.vec_fc(x_vec)
        comb_hidden = torch.cat([img_hidden, big_img_hidden, vec_hidden], dim=-1)
        comb_hidden = self.comb_layer(comb_hidden)

        policy_logits = self.policy_head(comb_hidden)

        value_in = comb_hidden.reshape(batch_size, -1)
        value = self.value_head(value_in)

        return policy_logits, value


class Agent(object):
    def __init__(self, use_gpu, *args, **kwargs):
        self.use_gpu = use_gpu

        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.state_handler_dict = {}

        torch.set_num_threads(1)
        self.training_iter = 0

    def register_model(self, name, model):
        assert isinstance(model, nn.Module)
        if name in self.state_handler_dict:
            raise KeyError(f"model named with {name} reassigned.")
        self.state_handler_dict[name] = model

    def loads(self, agent_dict):
        self.training_iter = agent_dict['training_iter']

        for name, np_dict in agent_dict['model_dict'].items():
            model = self.state_handler_dict[name]  # alias
            state_dict = {
                k: torch.as_tensor(v.copy(), device=self.device)
                for k, v in zip(model.state_dict().keys(), np_dict.values())
            }
            model.load_state_dict(state_dict)


def legal_mask(logit, legal):
    mask = torch.ones_like(legal) * -math.inf
    logit = torch.where(legal == 1., logit, mask)
    return logit


def one_hot_generator(n_feature, index):
    arr = np.zeros(n_feature,)
    arr[index] = 1
    return arr


def tensorize_state(func):
    def _recursive_processing(state, device):
        if not isinstance(state, torch.Tensor):
            if isinstance(state, dict):
                for k, v in state.items():
                    state[k] = _recursive_processing(state[k], device)
            else:
                state = torch.FloatTensor(state).to(device)
        return state

    @functools.wraps(func)
    def wrap(self, state, *arg, **kwargs):
        state = copy.deepcopy(state)
        state = _recursive_processing(state, self.device)
        return func(self, state, *arg, **kwargs)

    return wrap


class NMMOAgent(Agent):
    def __init__(self, use_gpu):
        super().__init__(use_gpu)
        self.net = NMMOModel().to(self.device)
        self.register_model('net', self.net)

    @tensorize_state
    def infer(self, state):
        with torch.no_grad():
            logits, value = self.net.infer(state)
            logits = legal_mask(logits, state['legal'])

            prob = F.softmax(logits, dim=-1).squeeze()
            action = prob.argmax(dim=-1).numpy()

        return action


class Translator:
    DUMMY_IMG = np.zeros((N_CH, WINDOW, WINDOW))
    ATTACK_AREA = np.zeros((WINDOW, WINDOW))

    def __init__(self):
        self.config = CompetitionConfig()

        self.ATTACK_AREA[WINDOW_CENTER-4: WINDOW_CENTER+5, WINDOW_CENTER-4: WINDOW_CENTER+5] = 1 / 3
        self.ATTACK_AREA[WINDOW_CENTER-3: WINDOW_CENTER+4, WINDOW_CENTER-3: WINDOW_CENTER+4] = 2 / 3
        self.ATTACK_AREA[WINDOW_CENTER-1: WINDOW_CENTER+2, WINDOW_CENTER-1: WINDOW_CENTER+2] = 1.

        self.obs = None
        self.team_dones = None
        self.time_step = None
        self.spawn_pos = None
        self.furthest_explore = None
        self.atk_targets_id = None

        self.last_obs = None
        self.last_12_obs = None
        self.last_8_move = None  # used for generating onehot feature
        self.last_32_pos = None  # used to draw on 2d-img feature
        self.last_atk_type = None
        self.last_atk_target = None
        self.last_atk_agent_id = None
        self.last_3_img = None
        self.max_food = None
        self.max_water = None
        self.max_health = None
        self.cum_dmg_recv = None  # cumulative damage received
        self.atk_type_cnt = None
        self.cum_dmg_dealt = None  # cumulative damage dealt for 3 styles
        self.cum_water_restored = None
        self.cum_food_restored = None
        self.kill_npc_num = None
        self.kill_player_num = None
        self.all_players_max_food = None
        self.all_players_max_water = None
        self.all_players_max_health = None
        self.passable_map = None
        self.pos_history = None

    def reset(self, init_obs):
        self.time_step = 0
        self.spawn_pos = [ob['Entity']['Continuous'][0, 5:7] for ob in init_obs.values()]
        self.furthest_explore = [0. for _ in range(N_PLAYER_PER_TEAM)]

        self.last_obs = init_obs
        self.last_12_obs = collections.deque([init_obs] * 12, maxlen=12)
        self.last_8_move = [collections.deque([one_hot_generator(N_MOVE_ACTION, -1)] * 8, maxlen=8)
                            for _ in range(N_PLAYER_PER_TEAM)]
        self.last_32_pos = [collections.deque([self.spawn_pos[i]] * 32, maxlen=32)
                            for i in range(N_PLAYER_PER_TEAM)]
        self.last_atk_type = [-1 for _ in range(N_PLAYER_PER_TEAM)]  # -1 denotes no atk
        self.last_atk_target = [-1 for _ in range(N_PLAYER_PER_TEAM)]  # -1 denotes no atk
        self.last_atk_agent_id = [0. for _ in range(N_PLAYER_PER_TEAM)]
        self.last_3_img = [collections.deque([np.zeros((N_CH, WINDOW, WINDOW))] * 3, maxlen=3)
                           for _ in range(N_PLAYER_PER_TEAM)]
        self.max_food = [10. for _ in range(N_PLAYER_PER_TEAM)]
        self.max_water = [10. for _ in range(N_PLAYER_PER_TEAM)]
        self.max_health = [10. for _ in range(N_PLAYER_PER_TEAM)]
        self.cum_dmg_recv = [0. for _ in range(N_PLAYER_PER_TEAM)]
        self.atk_type_cnt = [np.zeros((N_ATK_TYPE,)) for _ in range(N_PLAYER_PER_TEAM)]
        self.cum_dmg_dealt = [np.zeros((N_ATK_TYPE,)) for _ in range(N_PLAYER_PER_TEAM)]
        self.cum_water_restored = [0. for _ in range(N_PLAYER_PER_TEAM)]
        self.cum_food_restored = [0. for _ in range(N_PLAYER_PER_TEAM)]
        self.kill_npc_num = [0. for _ in range(N_PLAYER_PER_TEAM)]
        self.kill_player_num = [0. for _ in range(N_PLAYER_PER_TEAM)]
        self.all_players_max_food = {i+1: 10. for i in range(N_PLAYER)}
        self.all_players_max_water = {i+1: 10. for i in range(N_PLAYER)}
        self.all_players_max_health = {i+1: 10. for i in range(N_PLAYER)}

        # info for full-size feature map
        self.pos_history = [[] for _ in range(N_PLAYER_PER_TEAM)]
        self.passable_map = np.ones((TERRAIN_SIZE+1, TERRAIN_SIZE+1))
        self.passable_map[MAP_LEFT:MAP_RIGHT+1, MAP_LEFT:MAP_RIGHT+1] = 0.  # 1. stands for lava/water/stone

    def trans_obs(self, obs):
        self.obs = obs
        self.last_12_obs.append(obs)
        self.team_dones = {i: i not in obs for i in range(N_PLAYER_PER_TEAM)}

        def mark_point(arr_2d, index_arr, value, clip=False):
            arr_2d[index_arr[:, 0], index_arr[:, 1]] = \
                np.clip(value, 0., 1.) if clip else value

        player_path_map = np.zeros((N_PLAYER_PER_TEAM, TERRAIN_SIZE+1, TERRAIN_SIZE+1))

        # update inner status for each agent
        for i in range(N_PLAYER_PER_TEAM):
            if self.team_dones.get(i, True):
                continue

            # 0	 Mask          whether this row contains information, 1 for useful, 0 for null.
            # 1	 Entity_ID     ID of this entity; for players, ID>0, for NPCs, ID<0
            # 2	 Attacker_ID   the ID of last agent that attacks this entity
            # 3	 Level         the level of this entity
            # 4	 Population    the population this entity belongs to, can be used to identify teammates and opponents;
            #                  for players, population>=0; for NPCs, population<0
            # 5	 Row_index     the row index of this entity
            # 6	 Column_index  the column index of this entity
            # 7	 Damage        the damage this entity has been received
            # 8	 Timealive     the time this entity has been alive
            # 9	 Food          current food this entity has
            # 10 Water         current water this entity has
            # 11 Health        current health of this entity
            # 12 Is_freezed    whether this entity is frozen right now, 1 for frozen, 0 for not.

            # update the cumulative damage dealt for each atk style
            entity_obs = obs[i]['Entity']['Continuous']
            n_entity = int(sum(entity_obs[:, 0]))
            entity_obs = entity_obs[:n_entity]
            for row in entity_obs[1:]:
                if row[2] == entity_obs[0, 1]:  # attacker is me
                    self.cum_dmg_dealt[i][self.last_atk_type[i]] += row[7]

            # update the cumulative food/water restored
            last_entity_obs = self.last_obs[i]['Entity']['Continuous']
            restored_food = max(0, entity_obs[0, 9] - last_entity_obs[0, 9])
            restored_water = max(0, entity_obs[0, 10] - last_entity_obs[0, 10])
            self.cum_food_restored[i] += restored_food
            self.cum_water_restored[i] += restored_water

            # update shared info of all players' seen max food/water/health
            for row in entity_obs:
                if row[1] < 0:
                    continue
                self.all_players_max_food[row[1]] = max(self.all_players_max_food[row[1]], row[9])
                self.all_players_max_water[row[1]] = max(self.all_players_max_water[row[1]], row[10])
                self.all_players_max_health[row[1]] = max(self.all_players_max_health[row[1]], row[11])

            # update the kill number
            entity_in_sight = entity_obs[:, 1]
            if self.last_atk_agent_id[i] not in entity_in_sight:
                if self.last_atk_agent_id[i] > 0:
                    self.kill_player_num[i] += 1
                elif self.last_atk_agent_id[i] < 0:
                    self.kill_npc_num[i] += 1

            # update the furthest distance explored so far
            curr_pos = obs[i]['Entity']['Continuous'][0, 5:7]
            if self.spawn_pos[i][0] == 16.:
                explore = curr_pos[0] - 16.
            elif self.spawn_pos[i][0] == 144.:
                explore = 144. - curr_pos[0]
            elif self.spawn_pos[i][1] == 16.:
                explore = curr_pos[1] - 16.
            elif self.spawn_pos[i][1] == 144.:
                explore = 144. - curr_pos[1]
            else:
                raise RuntimeError('Unknown spawn position:', self.spawn_pos[i])
            self.furthest_explore[i] = max(self.furthest_explore[i], explore)

            # update the full-size feature map
            tile_obs = obs[i]['Tile']['Continuous']
            tile_pos = tile_obs[:, -2:].astype(int)
            obstacle_pos = np.logical_or(np.logical_or(tile_obs[:, 1] == 0, tile_obs[:, 1] == 1), tile_obs[:, 1] == 5)
            mark_point(self.passable_map, tile_pos, obstacle_pos)
            curr_pos = curr_pos.astype(int)
            self.pos_history[i].append(curr_pos)
            for k, pos in enumerate(self.pos_history[i]):
                player_path_map[i][pos[0], pos[1]] = (k + 1) / (self.time_step + 1)

        img = np.zeros((N_PLAYER_PER_TEAM, N_CH, WINDOW, WINDOW))
        for i in range(N_PLAYER_PER_TEAM):
            # replace with dummy feature if dead
            if self.team_dones.get(i, True):
                img[i] = self.DUMMY_IMG
                continue

            # 0	N_entity        the current number of entities on this tile
            # 1	Type            the type of this tile,
            #                   0 for lava, 1 for water, 2 for grass,
            #                   3 for scrub, 4 for forest, 5 for stone
            # 2	Row_index       the row index of this tile
            # 3	Column_index	the column index of this tile

            tile_obs = obs[i]['Tile']['Continuous']
            tile_pos = tile_obs[:, -2:].astype(int)
            tile_rel_pos = tile_pos - tile_pos[0, :]

            mark_point(img[i][CH_HAS_ENTITY], tile_rel_pos, tile_obs[:, 0])
            mark_point(img[i][CH_TILE_LAVA], tile_rel_pos, tile_obs[:, 1] == 0)
            mark_point(img[i][CH_TILE_WATER], tile_rel_pos, tile_obs[:, 1] == 1)
            mark_point(img[i][CH_TILE_GRASS], tile_rel_pos, tile_obs[:, 1] == 2)
            mark_point(img[i][CH_TILE_SCRUB], tile_rel_pos, tile_obs[:, 1] == 3)
            mark_point(img[i][CH_TILE_FOREST], tile_rel_pos, tile_obs[:, 1] == 4)
            mark_point(img[i][CH_TILE_STONE], tile_rel_pos, tile_obs[:, 1] == 5)
            mark_point(img[i][CH_POS_X], tile_rel_pos, tile_pos[:, 0] / TERRAIN_SIZE)
            mark_point(img[i][CH_POS_Y], tile_rel_pos, tile_pos[:, 1] / TERRAIN_SIZE)
            if self.spawn_pos[i][0] == 16.:
                spawn_dist_diff = tile_pos[:, 0] - 16.
            elif self.spawn_pos[i][0] == 144.:
                spawn_dist_diff = 144. - tile_pos[:, 0]
            elif self.spawn_pos[i][1] == 16.:
                spawn_dist_diff = tile_pos[:, 1] - 16.
            elif self.spawn_pos[i][1] == 144.:
                spawn_dist_diff = 144. - tile_pos[:, 1]
            else:
                raise RuntimeError('Unknown spawn position:', self.spawn_pos[i])
            mark_point(img[i][CH_SPAWN_DIST_DIFF], tile_rel_pos, spawn_dist_diff / MAP_SIZE)
            furthest_explore_diff = spawn_dist_diff - self.furthest_explore[i]
            mark_point(img[i][CH_FURTHEST_EXPLORE_DIFF], tile_rel_pos, furthest_explore_diff / MAP_SIZE)
            img[i][CH_ATTACK_AREA] = self.ATTACK_AREA

            # 0	 Mask          whether this row contains information, 1 for useful, 0 for null.
            # 1	 Entity_ID     ID of this entity; for players, ID>0, for NPCs, ID<0
            # 2	 Attacker_ID   the ID of last agent that attacks this entity
            # 3	 Level         the level of this entity
            # 4	 Population    the population this entity belongs to, can be used to identify teammates and opponents;
            #                  for players, population>=0; for NPCs, population<0
            # 5	 Row_index     the row index of this entity
            # 6	 Column_index  the column index of this entity
            # 7	 Damage        the damage this entity has been received
            # 8	 Timealive     the time this entity has been alive
            # 9	 Food          current food this entity has
            # 10 Water         current water this entity has
            # 11 Health        current health of this entity
            # 12 Is_freezed    whether this entity is frozen right now, 1 for frozen, 0 for not.

            entity_obs = obs[i]['Entity']['Continuous']
            n_entity = int(sum(entity_obs[:, 0]))
            entity_obs = entity_obs[:n_entity]
            entity_pos = entity_obs[:, 5:7].astype(int)
            entity_rel_pos = entity_pos - tile_pos[0, :]

            my_population = entity_obs[0, 4]
            entity_populations = entity_obs[:, 4]
            mark_point(img[i][CH_ENTITY_OUR], entity_rel_pos[entity_populations == my_population], 1.)
            entity_is_enemy = np.logical_and(entity_populations >= 0, entity_populations != my_population)
            mark_point(img[i][CH_ENTITY_ENEMY], entity_rel_pos[entity_is_enemy], 1.)
            mark_point(img[i][CH_ENTITY_NPC_PASSIVE], entity_rel_pos[entity_populations == -1], 1.)
            mark_point(img[i][CH_ENTITY_NPC_NEUTRAL], entity_rel_pos[entity_populations == -2], 1.)
            mark_point(img[i][CH_ENTITY_NPC_HOSTILE], entity_rel_pos[entity_populations == -3], 1.)
            mark_point(img[i][CH_ATTACKER_ME], entity_rel_pos, entity_obs[:, 2] == entity_obs[0, 1])
            attacker_populations = (entity_obs[:, 2] - 1) // N_PLAYER_PER_TEAM  # negative value still denotes npc
            mark_point(img[i][CH_ATTACKER_OUR], entity_rel_pos, attacker_populations == my_population)
            attacker_is_enemy = np.logical_and(attacker_populations >= 0, attacker_populations != my_population)
            mark_point(img[i][CH_ATTACKER_ENEMY], entity_rel_pos, attacker_is_enemy)
            mark_point(img[i][CH_ATTACKER_NPC], entity_rel_pos, attacker_populations < 0)
            mark_point(img[i][CH_LEVEL], entity_rel_pos, entity_obs[:, 3] / 50.)
            mark_point(img[i][CH_DAMAGE], entity_rel_pos, entity_obs[:, 7] / 10.)
            mark_point(img[i][CH_FOOD], entity_rel_pos, entity_obs[:, 9] / self.max_food[i])
            mark_point(img[i][CH_WATER], entity_rel_pos, entity_obs[:, 10] / self.max_water[i])
            mark_point(img[i][CH_HEALTH], entity_rel_pos, entity_obs[:, 11] / self.max_health[i])
            mark_point(img[i][CH_FROZEN], entity_rel_pos, entity_obs[:, 12])
            img[i][CH_PROGRESS] = self.time_step / MAX_STEP

            # mark history path
            last_32_rel_pos = (np.stack(self.last_32_pos[i]) - tile_pos[0, :]).astype(np.int)
            for k, pos in enumerate(last_32_rel_pos):
                if 0 <= pos[0] < WINDOW and 0 <= pos[1] < WINDOW:
                    # the value of newer point will replace the older
                    img[i][CH_HISTORY_PATH][pos[0], pos[1]] = (k + 1) / 32.

            self.last_3_img[i].append(img[i])
            self.last_32_pos[i].append(entity_pos[0])

        img = np.stack([
            np.concatenate([
                self.last_3_img[i][2][:],
                self.last_3_img[i][1][TO_STACK_CH],
                self.last_3_img[i][0][TO_STACK_CH],
            ])
            for i in range(N_PLAYER_PER_TEAM)
        ])

        self_vec = []
        target_vec = []
        ally_vec = []
        legal_atk = []
        atk_targets_id = []
        for i in range(N_PLAYER_PER_TEAM):
            if self.team_dones.get(i, True):
                self_vec.append(np.zeros((N_FEAT_SELF,)))
                target_vec.append(np.zeros((10 * N_FEAT_PER_TARGET,)))  # 10 target presented in state
                ally_vec.append(np.zeros((3 * N_FEAT_PER_ALLY,)))  # 3 allies considered
                legal_atk.append([0.] * N_ATK_ACTION)  # all atk actions are illegal
                atk_targets_id.append([-1.] * N_ATK_TARGET)  # dummy ids, should never be used
                continue

            # 0	 Mask          whether this row contains information, 1 for useful, 0 for null.
            # 1	 Entity_ID     ID of this entity; for players, ID>0, for NPCs, ID<0
            # 2	 Attacker_ID   the ID of last agent that attacks this entity
            # 3	 Level         the level of this entity
            # 4	 Population    the population this entity belongs to, can be used to identify teammates and opponents;
            #                  for players, population>=0; for NPCs, population<0
            # 5	 Row_index     the row index of this entity
            # 6	 Column_index  the column index of this entity
            # 7	 Damage        the damage this entity has been received
            # 8	 Timealive     the time this entity has been alive
            # 9	 Food          current food this entity has
            # 10 Water         current water this entity has
            # 11 Health        current health of this entity
            # 12 Is_freezed    whether this entity is frozen right now, 1 for frozen, 0 for not.

            entity_obs = obs[i]['Entity']['Continuous'].copy()
            entity_obs = entity_obs[entity_obs[:, 0] == 1.]  # remove the rows without information
            entity_obs = np.insert(entity_obs, entity_obs.shape[-1], list(range(len(entity_obs))), axis=-1)
            # 13 Index         the entity index which will be later used in action translation
            l1_dist = np.max(abs(entity_obs[:, 5:7] - entity_obs[0, 5:7]), axis=-1)
            entity_obs = np.insert(entity_obs, entity_obs.shape[-1], l1_dist, axis=-1)
            # 14 Distance      the L1 distance from self to target
            n_ally = np.array([sum(entity_obs[:, 4] == row[4]) - 1 if row[4] >= 0 else 0 for row in entity_obs])
            entity_obs = np.insert(entity_obs, entity_obs.shape[-1], n_ally, axis=-1)
            # 15 Num_allies    the number of allies of this entity in view

            def log_mapping(value):
                return np.log(1 + value) / 6.

            # figure out the nearest ally
            my_pos = obs[i]['Entity']['Continuous'][0, 5:7]
            allies_pos = [np.array([np.inf, np.inf]) for _ in range(N_PLAYER_PER_TEAM)]
            for j in range(N_PLAYER_PER_TEAM):
                if j != i and not self.team_dones.get(j, True):
                    allies_pos[j] = obs[j]['Entity']['Continuous'][0, 5:7]
            allies_pos_diff = np.sum(abs(np.array(allies_pos) - my_pos), axis=-1)
            nearest_allies_idx = [idx for idx in np.argsort(allies_pos_diff) if idx != i]
            allies_x_diff = [allies_pos[idx][0] - my_pos[0] for idx in nearest_allies_idx]
            allies_y_diff = [allies_pos[idx][1] - my_pos[1] for idx in nearest_allies_idx]
            allies_dist = [allies_pos_diff[idx] for idx in nearest_allies_idx]

            # extract features of controlled agent
            my_in_team_idx = (int(entity_obs[0, 1]) - 1) % 8
            my_food = entity_obs[0, 9]
            my_water = entity_obs[0, 10]
            my_health = entity_obs[0, 11]
            my_damage = entity_obs[0, 7]
            self.max_food[i] = max(self.max_food[i], my_food)
            self.max_water[i] = max(self.max_water[i], my_water)
            self.max_health[i] = max(self.max_health[i], my_health)
            my_last_12_info = [self.last_12_obs[k][i]['Entity']['Continuous'] for k in range(12)]
            my_last_12_food = [my_last_12_info[k][0, 9] for k in range(12)]
            my_last_12_water = [my_last_12_info[k][0, 10] for k in range(12)]
            my_last_12_health = [my_last_12_info[k][0, 11] for k in range(12)]
            my_last_12_dmg = [my_last_12_info[k][0, 7] for k in range(12)]
            self.cum_dmg_recv[i] += my_damage
            self_vec.append(np.array([
                *((my_pos - MAP_LEFT) / MAP_SIZE),
                *one_hot_generator(N_PLAYER_PER_TEAM, my_in_team_idx),
                *(np.arange(4) == my_food),  # near exhausted warning
                *(np.arange(4) == my_water),
                *(np.arange(4) == my_health),
                self.max_food[i] / 50.,
                self.max_water[i] / 50.,
                self.max_health[i] / 50.,
                (max(self.max_food) - self.max_food[i]) / 25.,  # diff to team's best
                (max(self.max_water) - self.max_water[i]) / 25.,
                (max(self.max_health) - self.max_health[i]) / 25.,
                *my_last_12_food,
                *my_last_12_water,
                *my_last_12_health,
                *my_last_12_dmg,
                log_mapping(self.cum_dmg_recv[i]),
                *np.concatenate(self.last_8_move[i]),
                *one_hot_generator(N_ATK_TYPE + 1, self.last_atk_type[i]),
                *one_hot_generator(N_ATK_TARGET + 1, self.last_atk_target[i]),
                *[log_mapping(self.atk_type_cnt[i][j]) for j in range(N_ATK_TYPE)],
                *[log_mapping(self.cum_dmg_dealt[i][j]) for j in range(N_ATK_TYPE)],
                *(np.clip(allies_x_diff, -30, 30) / 30),  # consider allies with 30 L1 distance
                *(np.clip(allies_y_diff, -30, 30) / 30),
                *(np.clip(allies_dist, -60, 60) / 60),
                sum(self.team_dones.values()) / N_PLAYER_PER_TEAM,  # num dead players in team
                self.furthest_explore[i] / MAP_SIZE,
                max(self.furthest_explore) / MAP_SIZE,
                self.furthest_explore[i] == max(self.furthest_explore),
                (max(self.furthest_explore) - self.furthest_explore[i]) / MAP_SIZE,
                max(self.furthest_explore) >= 127,
                (127 - max(self.furthest_explore)) / MAP_SIZE,
                max(self.furthest_explore) == 126,  # nearly approaching flag
                max(self.furthest_explore) == 125,
                max(self.furthest_explore) == 124,
                self.cum_food_restored[i] / 500.,
                self.cum_water_restored[i] / 500.,
                self.kill_npc_num[i] / 40.,
                self.kill_player_num[i] / 6.,
                max(self.kill_player_num) / 6.,
                self.kill_player_num[i] == max(self.kill_player_num),
                (max(self.kill_player_num) - self.kill_player_num[i]) / 6.,
                max(self.kill_player_num) == 1.,
                max(self.kill_player_num) == 2.,
                max(self.kill_player_num) == 3.,
                max(self.kill_player_num) == 4.,
                max(self.kill_player_num) == 5.,
                max(self.kill_player_num) >= 6.,
                self.kill_player_num[i] == 1.,
                self.kill_player_num[i] == 2.,
                self.kill_player_num[i] == 3.,
                self.kill_player_num[i] == 4.,
                self.kill_player_num[i] == 5.,
                self.kill_player_num[i] >= 6.,
            ]))

            def extract_target_feature(target):
                mask_arr = np.array([
                    target[14] <= 1,  # melee legal
                    target[14] <= 3,  # range legal
                    target[14] <= 4,  # mage legal
                ], dtype=np.float32)
                attacker_population = (target[2] - 1) // N_PLAYER_PER_TEAM  # negative value still denotes npc
                feat_arr = np.array([
                    target[2] == 0.,  # not be attacked
                    target[2] == entity_obs[0, 1],  # attacked by me
                    attacker_population == entity_obs[0, 4],  # attacked by my team
                    attacker_population >= 0 and attacker_population != entity_obs[0, 4],  # attacked by enemy
                    attacker_population < 0,  # attacked by npc
                    *(one_hot_generator(N_TEAM, int(target[4])) if target[4] >= 0 else np.zeros((N_TEAM,))),
                    target[1] == self.last_atk_agent_id[i],  # is the last attack's target
                    target[3] / 50.,  # level
                    (target[3] - entity_obs[0, 3]) / 25.,  # level diff
                    target[3] >= entity_obs[0, 3],  # higher level flag
                    (target[5] - entity_obs[0, 5]) / 4.,  # x diff
                    (target[6] - entity_obs[0, 6]) / 4.,  # y diff
                    target[7] / 10.,  # damage received
                    target[9] / 50.,  # food
                    target[10] / 50.,  # water
                    target[11] / 50.,  # health
                    self.all_players_max_food[target[1]] / 50. if target[1] > 0 else 0.,  # max food ever seen
                    self.all_players_max_water[target[1]] / 50. if target[1] > 0 else 0.,  # max water ever seen
                    self.all_players_max_health[target[1]] / 50. if target[1] > 0 else 0.,  # max health ever seen
                    target[9] / self.max_food[i],  # rel food
                    target[10] / self.max_water[i],  # rel water
                    target[11] / self.max_health[i],  # rel health
                    (target[11] - entity_obs[0, 11]) / 25.,  # health diff
                    target[11] >= entity_obs[0, 11],  # higher health flag
                    target[12],  # frozen
                    target[14] / 4.,  # l1 distance
                    target[15] / 2.,  # num allies in view
                    *mask_arr,  # legal mask of atk type
                ])
                return feat_arr, mask_arr

            # extract features of atk targets
            targets_feature = []
            atk_mask = []  # legal masks of the attacking actions of this agent, i.e. [ent1_melee, ent1_range, ent1_mage, ..., ent4_mage, no_attack]
            atk_tid = []  # the attacking action's corresponding target entity index, i.e. [ent1_idx, ent2_idx, ...]

            targets = entity_obs[entity_obs[:, 4] != entity_obs[0, 4]]  # with diff population
            player_targets = targets[targets[:, 4] >= 0]
            player_targets = player_targets[np.argsort(player_targets[:, 14])]  # sort by distance
            npc_targets = targets[targets[:, 4] < 0]
            npc_targets = npc_targets[np.argsort(npc_targets[:, 14])]
            for target_group, considered_n_nearest in zip([player_targets, npc_targets], [4, 6]):
                # here we present `considered_n_nearest` entities in state for each group,
                # but just choose the 2 nearest as the target of attack action
                for k in range(considered_n_nearest):
                    if len(target_group) > k:
                        target_ent = target_group[k]
                        feat, mask = extract_target_feature(target_ent)
                        if k < 2:
                            atk_mask += mask.tolist()
                            atk_tid.append(target_ent[13])
                    else:
                        feat = np.zeros((N_FEAT_PER_TARGET,))
                        if k < 2:
                            atk_mask += [0., 0., 0.]
                            atk_tid.append(-1.)
                    targets_feature.append(feat)
            # at least choose one target to attack, unless no entity in range
            atk_mask.append(float(sum(atk_mask) == 0))

            def extract_ally_feature(ally):
                attacker_population = (ally[2] - 1) // N_PLAYER_PER_TEAM  # negative value still denotes npc
                feat_arr = np.array([
                    ally[2] == 0.,  # not be attacked
                    attacker_population >= 0 and attacker_population != entity_obs[0, 4],  # attacked by enemy
                    attacker_population < 0,  # attacked by npc
                    ally[3] / 50.,  # level
                    (ally[3] - entity_obs[0, 3]) / 25.,  # level diff
                    ally[3] >= entity_obs[0, 3],  # higher level flag
                    (ally[5] - entity_obs[0, 5]) / 4.,  # x diff
                    (ally[6] - entity_obs[0, 6]) / 4.,  # y diff
                    ally[7] / 10.,  # damage received
                    ally[9] / 50.,  # food
                    ally[10] / 50.,  # water
                    ally[11] / 50.,  # health
                    self.all_players_max_food[ally[1]] / 50.,  # max food ever seen
                    self.all_players_max_water[ally[1]] / 50.,  # max water ever seen
                    self.all_players_max_health[ally[1]] / 50.,  # max health ever seen
                    ally[9] / self.all_players_max_food[ally[1]],  # rel food
                    ally[10] / self.all_players_max_water[ally[1]],  # rel water
                    ally[11] / self.all_players_max_health[ally[1]],  # rel health
                    (ally[11] - entity_obs[0, 11]) / 25.,  # health diff
                    ally[11] >= entity_obs[0, 11],  # higher health flag
                    ally[12],  # frozen
                    ally[14] / 4.,  # l1 distance
                    ally[15] / 2.,  # num allies in view
                ])
                return feat_arr

            # extract features of allies
            allies_feature = []

            allies = entity_obs[entity_obs[:, 4] == entity_obs[0, 4]][1:]  # with the same population, and is not itself
            allies = allies[np.argsort(allies[:, 14])]   # sort by distance
            for k in range(3):  # consider 3 nearest allies
                if len(allies) > k:
                    feat = extract_ally_feature(allies[k])
                else:
                    feat = np.zeros((N_FEAT_PER_ALLY,))
                allies_feature.append(feat)

            target_vec.append(np.concatenate(targets_feature))
            ally_vec.append(np.concatenate(allies_feature))
            legal_atk.append(atk_mask)
            atk_targets_id.append(atk_tid)

        self_vec = np.stack(self_vec)
        target_vec = np.stack(target_vec)
        ally_vec = np.stack(ally_vec)
        vec = np.concatenate([self_vec, target_vec, ally_vec], axis=-1)
        legal_atk = np.stack(legal_atk)
        self.atk_targets_id = np.stack(atk_targets_id)

        legal_move = np.ones([N_PLAYER_PER_TEAM, N_MOVE_ACTION])
        legal_die = np.zeros([N_PLAYER_PER_TEAM, 1])  # one special action, legal when dead
        for i in range(N_PLAYER_PER_TEAM):
            if self.team_dones.get(i, True):
                legal_move[i] = 0.
                legal_die[i] = 1.
                continue

            tiles_type = obs[i]['Tile']['Continuous'][:, 1].reshape((WINDOW, WINDOW))
            center = np.array([WINDOW_CENTER, WINDOW_CENTER])
            for action_idx, d in enumerate(Direction.edges):
                pos_after = center + d.delta
                if tiles_type[pos_after[0], pos_after[1]] in (0., 1., 5.):  # lava, water and stone
                    legal_move[i, action_idx] = 0.

            legal_move[i][-1] = 0.  # we finally decide to force it to move!

        legal = np.zeros([N_PLAYER_PER_TEAM, N_MOVE_ACTION, N_ATK_ACTION])
        for i in range(N_MOVE_ACTION):
            for j in range(N_ATK_ACTION):
                legal[:, i, j] = np.logical_and(legal_move[:, i], legal_atk[:, j])
        legal = legal.reshape(N_PLAYER_PER_TEAM, -1)
        legal = np.concatenate([legal, legal_die], axis=-1)

        # get each agent's passable/path
        passable_map = []
        path_map = []
        for i in range(N_PLAYER_PER_TEAM):
            if self.team_dones.get(i, True):
                passable_map.append(np.zeros((33, 33)))
                path_map.append(np.zeros((33, 33)))
                continue

            curr_pos = obs[i]['Entity']['Continuous'][0, 5:7].astype(int)
            left, right = curr_pos[0] - 16, curr_pos[0] + 16 + 1
            up, down = curr_pos[1] - 16, curr_pos[1] + 16 + 1
            passable_map.append(self.passable_map[left:right, up:down])
            path_map.append(player_path_map[i][left:right, up:down])
        passable_map = np.stack(passable_map, axis=0)
        path_map = np.stack(path_map, axis=0)
        big_map = np.stack([passable_map, path_map], axis=1)

        obs = {
            'big_img': big_map,
            'img': img,
            'vec': vec,
            'legal': legal,
        }
        self.last_obs = self.obs
        return obs

    def trans_action(self, raw_actions):
        self.time_step += 1

        actions = {}
        for i in range(N_PLAYER_PER_TEAM):
            raw_act = raw_actions[i]
            if self.team_dones.get(i, True):
                assert raw_act == N_ACTION - 1
                continue

            actions[i] = {}

            move_dir = raw_act // N_ATK_ACTION
            self.last_8_move[i].append(one_hot_generator(N_MOVE_ACTION, move_dir))
            if move_dir != N_MOVE_ACTION - 1:  # is not idle
                actions[i][nmmo.io.action.Move] = {
                    nmmo.io.action.Direction: move_dir,
                }

            atk = raw_act % N_ATK_ACTION
            if atk != N_ATK_ACTION - 1:  # is not idle
                atk_target = int(self.atk_targets_id[i][atk // N_ATK_TYPE])
                atk_style = atk % N_ATK_TYPE
                self.last_atk_agent_id[i] = self.obs[i]['Entity']['Continuous'][atk_target][1]
                actions[i][nmmo.io.action.Attack] = {
                    nmmo.io.action.Style: atk_style,
                    nmmo.io.action.Target: atk_target,
                }
                self.last_atk_type[i] = atk_style
                self.last_atk_target[i] = atk // N_ATK_TYPE
                self.atk_type_cnt[i][atk_style] += 1
            else:
                self.last_atk_agent_id[i] = 0
                self.last_atk_type[i] = -1
                self.last_atk_target[i] = -1

        return actions


class AI(Team):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = NMMOAgent(False)
        model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'model.pth')
        agent_dict = torch.load(model_path)
        self.agent.loads(agent_dict)

        self.translator = Translator()
        self.step = 0

    def act(self, observations):
        if self.step == 0:
            self.translator.reset(observations)
        state = self.translator.trans_obs(observations)
        actions = self.agent.infer(state)
        actions = self.translator.trans_action(actions)
        self.step += 1
        return actions


class Submission:
    team_klass = AI
    init_params = {}
