# differentiable_discrete_communication

Code for Communication Learning via Backpropagation in Discrete Channels with Unknown Noise

Paper: https://ojs.aaai.org/index.php/AAAI/article/view/6205

File list:
-ACNet_comms_hidden_agent_bitstring_credit_assignment.py
  -Implements actor and critic networks for reinforced communication learning for hidden-goal navigation problem
-ACNet_diff_comms_recon_error.py
  -Implements actor and critic networks for continuous communication learning for hidden-goal navigation problem
-ACNet_diff_comms_recon_error_len10.py	
  -Implements actor and critic networks for discrete differentiable communication learning for hidden-goal navigation problem
-ACNet_rl_search2.py
  -Implements actor and critic network for reinforced communication learning for search problem
-ACNet_search_diff_comms_recon_error_team.py
  -Implements actor and critic networks for continuous communication learning for multi-agent search problem
-ACNet_search_diff_comms_recon_error_circle.py		
  -Implements actor and critic networks for discrete differentiable communication learning for hidden-goal navigation problem with channel noise

-diff_continuous_mapf.py
  -Trains agent with continuous communication for hidden-goal navigation problem
-diff_continuous_search.py
  -Trains agent with continuous communication for multi-agent search problem
-diff_discrete_mapf.py
  -Trains agent with discrete differentiable communication for hidden-goal navigation problem
-diff_discrete_mapf_noise.py
   -Trains agent with discrete differentiable communication for hidden-goal navigation problem with channel noise
-diff_discrete_search.py		
  -Trains agent with discrete differentiable communication for multi-agent search problem
-diff_discrete_search_noise.py
  -Trains agent with discrete differentiable communication for multi-agent search problem with channel noise

-GroupLock.py
  -Impelemnts locks for multi-thread code

-mapf_gym_search_diff_comms.py
  -Environment for multi-agent search with differentiable communications
-mapf_gym_diff_comms.py
  -Environment for hidden-goal navigation problem with differentiable communications
-mapf_gym_hidden_agent_bitstring_credit_assignment.py
  -Environment for hidden-goal navigation problem with reinforced communication learned
-mapf_gym_rl_search.py
  -Environment for multi-agent search problem with reinforced communication learned 

-req.txt
  -installed packages
-rl_comms_mapf.py
  -Trains agents with reinforced communication learning on hidden-goal navigation problem
-rl_comms_mapf_noise.py
  -Trains agents with reinforced communication learning on hidden-goal navigation problem with channel noise
-rl_comms_search.py
  -Trains agents with reinforced communication learning on multi-agent search problem
-rl_comms_search_noise.py
  -Trains agents with reinforced communication learning on multi-agent search problem with channel noise
