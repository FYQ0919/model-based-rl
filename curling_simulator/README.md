# curling\_simulator

冰壶仿真环境

## 重要提示
- 更新taichi至最新版本 (>=1.0.1)! 
- 目前该仓库代码已经改写为python module, 可以作为其他项目的module, 所以无法直接运行
    - 运行方式举例：
    - `python -m curling_simulator.gym_env`
    - `python -m curling_simulator.curling_env`
    - ipython中: `%run -m curling_simulator.gym_env`
    - ipython中: `%run -m curling_simulator.curling_env`


## 后续任务 TODO:

@env
- (目前)完善多智能体版本的gym环境
- 调整环境obs设置
  - image obs
    - player A 球
    - player B 球
    - 一共还剩几个球(A+B)
      - A/B 
      - 剩几轮
  - 图片化
    - 离散化:归属一个格子:0/1
      - 韩国ICML\sci. robots论文
    - 连续化:
  - history obs
- 调整出手位置 pos_x, pos_y 的上下界
  - pos_x, pos_y, vel_x, vel_y 之间可能存在关联

@algorithm
- (目前) 套用muzero - continuous版本算法, 训练curling AI
- muzero调参
- 寻找&阅读muzero continuous的实现


暂时不要使用taichi cuda后端, 否则pytorch神经网络loss backward时会出现以下错误（猜测是taichi的问题）:
```
Traceback (most recent call last):
  File "train_curling.py", line 233, in <module>
    exp_manager.learn(model)
  File "/home/codenie/curlingAI/curlingAI_SB3zoo/utils/exp_manager.py", line 218, in learn
    model.learn(self.n_timesteps, **kwargs)
  File "/home/codenie/.conda/envs/curling/lib/python3.8/site-packages/stable_baselines3/sac/sac.py", line 292, in learn
    return super(SAC, self).learn(
  File "/home/codenie/.conda/envs/curling/lib/python3.8/site-packages/stable_baselines3/common/off_policy_algorithm.py", line 366, in learn
    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
  File "/home/codenie/.conda/envs/curling/lib/python3.8/site-packages/stable_baselines3/sac/sac.py", line 226, in train
    ent_coef_loss.backward()
  File "/home/codenie/.conda/envs/curling/lib/python3.8/site-packages/torch/_tensor.py", line 363, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
  File "/home/codenie/.conda/envs/curling/lib/python3.8/site-packages/torch/autograd/__init__.py", line 173, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: Event device type CUDA does not match blocking stream's device type CPU.
```