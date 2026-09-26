
# ThunderAgent origin
nohup thunderagent --backend-type vllm --metrics \
> ./serve_thunderagent.log &

# Oracle
nohup thunderagent --backend-type vllm --metrics \
--scheduler-policy remaining_steps \
--dynamic-sd --sd-switch-threshold 64 \
> ./serve_thunderagent.log &

# SpecSys Schecule
nohup thunderagent --backend-type vllm --metrics \
--scheduler-policy predicted_remaining_steps \
> ./serve_thunderagent.log &