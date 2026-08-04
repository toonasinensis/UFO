
remote_path1="xiechunyang@112.74.164.141:/home/xiechunyang/wt_ws/wt_wbc/"
local_path1="/home/thl/wt_wbc/UFO"
exclude_folders1=(".vscode" 
                  "logs" 
                  ".git"
                  "output"
                  "run"
                  ".venv"
                  "results"
                  "_wandb"
                  "__pycache__"
                  "*.pyc"
                  ) 

# 构建排除参数
exclude_args=""
for folder in "${exclude_folders1[@]}"; do
    exclude_args="$exclude_args --exclude=$folder"
done

# 执行 rsync 命令（指定端口 9765）
rsync -avz $exclude_args -e "ssh -p 22" $local_path1 $remote_path1
#ssh 密码是Lejurobot2026
