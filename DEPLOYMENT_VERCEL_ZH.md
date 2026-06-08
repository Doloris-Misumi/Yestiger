# YesTiger 上线方案：GitHub + Vercel

## 结论

可以用 GitHub + Vercel 快速上线一个不用买域名的网站。Vercel 会自动给项目分配一个 `*.vercel.app` 地址。

但当前 YesTiger 的“上传任意歌曲并分析”后端依赖 Python、音频解码和较大的 MP3 上传。Vercel Function 的请求体限制不适合直接承载这部分。因此推荐拆成两层：

- Vercel：部署网页、静态示例、视频预览与导出界面。
- Python 后端：部署 `webapp/server.py`，负责上传音频、分析、生成 callbook。

现在仓库已经支持这种拆分。没有后端时，Vercel 页面仍能加载 14 首静态示例；接上后端后，就能分析上传歌曲。

## 方案 A：先上线前端演示站

1. 把当前仓库推到 GitHub。
2. 打开 Vercel Dashboard，选择 `New Project`。
3. Import 这个 GitHub 仓库。
4. Framework Preset 选 `Other` 或让 Vercel 自动识别。
5. Build Command 留空。
6. Output Directory 留空。
7. 点击 Deploy。

部署完成后，Vercel 会给一个类似下面的地址：

```text
https://yetiger.vercel.app
```

这个版本可以做：

- 浏览 14 首静态示例。
- 查看动作时间线。
- 修改动作名称。
- 导出 JSON / Markdown。
- 浏览器内录制 WebM。

这个版本暂时不能做：

- 在线上传一首新歌并云端分析。

## 方案 B：接入 Python 分析后端

把 `webapp/server.py` 部署到一个支持较大上传体积和 Python 音频依赖的平台，例如 Render、Railway、Fly.io、Hugging Face Spaces 或一台自己的服务器。

后端启动命令：

```powershell
.\.venv\Scripts\python.exe webapp\server.py --host 0.0.0.0 --port 8765
```

假设后端地址是：

```text
https://your-yetiger-api.example.com
```

有两种方式让前端连接它。

方式 1：打开页面时带参数：

```text
https://yetiger.vercel.app?api=https://your-yetiger-api.example.com
```

方式 2：修改 [webapp/static/config.js](D:/yetiger/webapp/static/config.js)：

```javascript
window.YESTIGER_API_BASE = "https://your-yetiger-api.example.com";
```

然后提交到 GitHub，Vercel 会自动重新部署。

## 为什么不直接把全部东西塞进 Vercel

Vercel 可以部署 Python Function，但 YesTiger 现在不适合全放进去：

- 一首 MP3 往往超过 4.5 MB，请求体会碰到限制。
- `torch/torchaudio` 一类依赖很重，Function 打包和冷启动都不理想。
- 音频分析更适合放在能持续运行的 Python 服务里。

所以短期最稳的汇报/展示路线是：

1. Vercel 上线前端，让老师和同学能打开网页看。
2. 云端后端或本机后端负责真实上传分析。
3. 后续再把后端容器化，做成正式服务。

## 已加入的部署文件

- [vercel.json](D:/yetiger/vercel.json)：Vercel 路由配置。
- [webapp/static/config.js](D:/yetiger/webapp/static/config.js)：线上 API 地址配置。
- [webapp/static/examples](D:/yetiger/webapp/static/examples)：静态示例数据。
