import torch
from torch.nn import MSELoss


def loss_func(batch, net, tgt_net, GAMMA, device):
    states, next_states, actions, rewards, dones = batch  

    state_v = torch.FloatTensor(states).to(device)

    next_state_v = torch.FloatTensor(next_states).to(device)

    actions_v = torch.LongTensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
       
    # Предсказание обучаемой сети
    state_action_values = net(state_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    
    # Предсказание обученной сети
    next_state_values = tgt_net(next_state_v).max(1)[0]  # Вернет значения лучших Q
    next_state_values[done_mask] = 0.0  # Для завершившихся эпизодов следующее состояние имеет значение 0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v  # Ожидаемое значение Q согласно ур-ю Белмана

    return MSELoss()(state_action_values, expected_state_action_values)