import torch
from torch.nn import MSELoss

def loss_func(batch, net, tgt_net, GAMMA, device):
    states, next_states, actions, rewards, dones = batch  
    
    states_l = states[0] / 255
    states_r = states[1] / 255
    next_states_l = next_states[0] / 255
    next_states_r = next_states[1] / 255
    
    state_v_l = torch.FloatTensor(states_l).to(device)
    state_v_r = torch.FloatTensor(states_r).to(device)
    
    next_state_v_l = torch.FloatTensor(next_states_l).to(device)
    next_state_v_r = torch.FloatTensor(next_states_r).to(device)
    
    actions_v = torch.LongTensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
       
    #Предсказание обучаемой сети
    state_action_values = net(state_v_l, state_v_r).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    
    #Предсказание обученной сети
    next_state_values = tgt_net(next_state_v_l, next_state_v_r).max(1)[0] #Вернет значения лучших Q
    next_state_values[done_mask] = 0.0 #Для завершившихся эпизодов следующее состояние имеет значение 0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v #Ожидаемое значение Q согласно уравнению Белмана


    return MSELoss()(state_action_values, expected_state_action_values)



# def loss_func(batch, net, tgt_net, GAMMA, device):
#     """
#     The method calculate MSE between predicted and real Q-values 
#     for the given state.

#     Parameters
#     ----------
#     batch : tuple
#         tuple of states, next_states, actions, rewards, dones.
#     net : instance of class Net
#     tgt_net : instance of class Net
#     GAMMA : float
#     device : string
#         Defines device for torch operations: cpu or cuda.

#     Returns
#     -------
#     torch.tensor
        
#     """
#     states, next_states, actions, rewards, dones = batch  
    
#     states_l = states[0] / 255
#     states_r = states[1] / 255
#     next_states_l = next_states[0] / 255
#     next_states_r = next_states[1] / 255
    
#     state_v_l = torch.FloatTensor(states_l).to(device)
#     state_v_r = torch.FloatTensor(states_r).to(device)
    
#     next_state_v_l = torch.FloatTensor(next_states_l).to(device)
#     next_state_v_r = torch.FloatTensor(next_states_r).to(device)
    
#     actions_v = torch.LongTensor(actions).to(device)
#     rewards_v = torch.FloatTensor(rewards).to(device)
#     done_mask = torch.BoolTensor(dones).to(device)
       
#     #Предсказание обучаемой сети
#     state_action_values = net(state_v_l, state_v_r).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    
#     #Предсказание обученной сети
#     next_state_values = tgt_net(next_state_v_l, next_state_v_r).max(1)[0] #Вернет значения лучших Q
#     next_state_values[done_mask] = 0.0 #Для завершившихся эпизодов следующее состояние имеет значение 0
#     next_state_values = next_state_values.detach()

#     expected_state_action_values = next_state_values * GAMMA + rewards_v #Ожидаемое значение Q согласно уравнению Белмана


#     return MSELoss()(state_action_values, expected_state_action_values)