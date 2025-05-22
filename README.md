# 安装依赖

```
git clone -b sxy/multimodal-rag https://github.com/JingofXin/LazyLLM.git

cd LazyLLM

pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple

```

```
pip install -r xxx.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```
curl -L -O https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install flash_attn-2.7.0.post2+cu12torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --no-deps --user --force-reinstall

pip install -v opencv-python==4.6.0.66 --no-deps --user --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -v opencv-contrib-python==4.6.0.66 --no-deps --user --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -v opencv-python-headless==4.6.0.66 --no-deps --user --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -v colpali-engine==0.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -v triton==3.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install transformers==4.45.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# 配置文件

为避免一些报错，启动应用前需设置好如下环境变量：
```
export LAZYLLM_MODEL_SOURCE=huggingface
```
由于使用了一些线上的大模型，所以需要配置好key:
```
export LAZYLLM_SENSENOVA_API_KEY=xxx
export LAZYLLM_SENSENOVA_SECRET_KEY=xxxx
export LAZYLLM_QWEN_API_KEY=xxx
```

# VS CODE远程调试的服务器无法连接huggingface和github怎么办

我的MAC本身可以连接huggingface和github，通过vscode让远程调试的服务器通过我的MAC来下载文件。        
这种技术通常称为 **SSH Remote Port Forwarding** (或反向隧道)。        

它的原理是：        
1.  你在你的MAC上运行一个简单的HTTP/HTTPS代理服务（或者利用SSH的SOCKS代理能力）。        
2.  当你从MAC通过SSH连接到远程服务器时（VS Code的远程连接就是基于SSH的），你指示SSH在远程服务器上打开一个端口。            
3.  任何发送到远程服务器上这个特定端口的流量，都会通过SSH隧道安全地转发回你的MAC。            
4.  你的MAC上的代理服务接收到这个流量，然后代表远程服务器去访问互联网（如Hugging Face）。           


**在MAC上运行一个轻量级HTTP代理 + SSH远程端口转发 (推荐)**

这种方法更明确，因为你清楚地在Mac上运行了一个代理。

**步骤 1: 在你的MAC上安装并运行一个HTTP代理**

你可以使用 `tinyproxy`，它非常轻量级。
*   **安装 (使用 Homebrew):**
    ```bash
    brew install tinyproxy
    ```
*   **配置 `tinyproxy`:**
    编辑配置文件，通常位于 `/usr/local/etc/tinyproxy/tinyproxy.conf` (macOS Intel) 或 `/opt/homebrew/etc/tinyproxy/tinyproxy.conf` (macOS Apple Silicon)。
    ```bash
    sudo nano /opt/homebrew/etc/tinyproxy/tinyproxy.conf # (或相应路径)
    ```
    确保以下设置 (或类似设置):
    ```
    Port 8888             # 代理监听的端口，你可以选择其他未被占用的端口
    Listen 127.0.0.1      # 只允许本地连接 (SSH隧道会从本地连接过来)
    Allow 127.0.0.1       # 允许来自本地的请求
    # DisableViaHeader Yes # 可选，隐藏Via头
    # ConnectPort 443     # 允许HTTPS的CONNECT方法
    # ConnectPort 563     # (如果需要其他HTTPS端口)
    ```
    保存并退出。
*   **启动 `tinyproxy`:**
    ```bash
    brew services start tinyproxy
    ```
    或者手动启动:
    ```bash
    tinyproxy -c /opt/homebrew/etc/tinyproxy/tinyproxy.conf
    ```
    验证它是否在运行并监听 `127.0.0.1:8888`:
    ```bash
    lsof -i :8888
    ```

**步骤 2: 配置SSH进行远程端口转发**

当VS Code连接到远程服务器时，它会使用你的SSH配置。你需要告诉SSH在远程服务器上打开一个端口，并将该端口的流量转发回你Mac上`tinyproxy`正在监听的 `127.0.0.1:8888`。

*   **编辑你MAC上的 `~/.ssh/config` 文件:**
    ```
    Host remote-server-7023  # 这是你在VS Code中连接时使用的主机名，请不要用IP:PORT的格式，会导致配置失效
        HostName actual_remote_server_ip_or_hostname
        User your_remote_username
        # IdentityFile ~/.ssh/your_private_key # 如果你使用密钥对

        # 远程端口转发:
        # RemoteForward <remote_port_on_server> <local_ip_on_mac_to_forward_to>:<local_port_on_mac_to_forward_to>
        # 在远程服务器上监听 6060 端口，并将流量转发到你Mac的 127.0.0.1:8888 (tinyproxy)
        RemoteForward 6060 127.0.0.1:8888
    ```
    *   `your_remote_server_alias`: 你在VS Code中连接时使用的主机名（例如，你在VS Code的SSH Targets中看到的名称）。
    *   `6060`: 这是远程服务器上将要监听的端口。你可以选择任何未被远程服务器占用的端口。
    *   `127.0.0.1:8888`: 这是你Mac上`tinyproxy`监听的地址和端口。

**步骤 3: 在远程服务器上设置代理环境变量**

现在，当VS Code通过SSH连接到远程服务器后，远程服务器上会有一个 `localhost:6060` 的端口，其流量会被转发到你Mac的代理。

*   **在远程服务器的终端中 (通过VS Code打开的终端即可):**
    ```bash
    export LAZYLM_HTTP_PROXY="http://localhost:6060"
    # 为了通用性，也设置标准的代理变量
    export http_proxy="http://localhost:6060"
    export https_proxy="http://localhost:6060" # 注意：https_proxy的值通常也是http://开头的代理地址

    # 可选：如果你有一些内部地址不想通过代理
    # export no_proxy="localhost,127.0.0.1,your_internal_domain.com"
    ```
    你可以将这些 `export` 命令添加到远程服务器的 `~/.bashrc` 或 `~/.zshrc` 文件中，使其永久生效。添加后，执行 `source ~/.bashrc` 或重新打开终端。

**步骤 4: 测试**

1.  确保 `tinyproxy` 在你的MAC上运行。
2.  通过VS Code连接到你的远程服务器 (它会自动使用 `~/.ssh/config` 中的 `RemoteForward` 设置)。
3.  在远程服务器的VS Code终端中，设置好代理环境变量。
4.  尝试运行你的Python脚本，或者简单地用 `curl` 测试：
    ```bash
    curl -v https://huggingface.co
    ```
    在 `curl` 的输出中，你应该能看到它尝试连接到 `localhost:6060`，然后通过代理成功获取内容。

**工作流程总结:**
1.  **Python script (on remote server)** tries to download from Hugging Face.
2.  It uses `LAZYLM_HTTP_PROXY` (or `http_proxy`/`https_proxy`) which is `http://localhost:6060`.
3.  Request goes to `localhost:6060` **on the remote server**.
4.  SSH `RemoteForward` intercepts this traffic and sends it through the SSH tunnel back to your **MAC's** `127.0.0.1:8888`.
5.  `tinyproxy` (running on your MAC) receives the request on port `8888`.
6.  `tinyproxy` makes the actual request to Hugging Face using your MAC's internet connection.
7.  Response travels back through the same path.

这种方法非常有效，并且VS Code的SSH集成使得 `RemoteForward` 的配置相对容易。记得在VS Code断开并重新连接远程服务器后，`RemoteForward` 才会生效。