# 可靠提交 Codex 决策

在完整读取本轮请求后，由当前 Codex 编写决策 JSON。推荐先将它保存为本地 UTF-8 文件，使用 SCP 原样上传到本次任务的唯一远程文件，再通过一次独立的 `ssh -T` 调用官方 `respond --response <远程文件>`。本地文件只是临时传输副本，正式决策及其状态留在服务器。

提交工具使用普通管道（例如 `exec_command` 的 `tty: false`），不要把长 JSON 输入到持续运行的 Windows SSH/ConPTY 交互控制台。控制台回显只说明输入被显示，不能证明远端收到。SSH 的 `-T` 禁用远端伪终端；本地工具也必须禁用 TTY。JSON 不进入 shell 命令字符串。无需本地 Python。

## Windows PowerShell

下面在本地 PowerShell 执行，只负责传输。将已由 Codex 写好的 UTF-8 JSON 文件和已确认的服务器路径填入变量；不是生成 planner/supervisor 答案的模板。示例远程路径只含 ASCII 字母、数字、`/._-`；其他路径须按远程 shell 正确引用。

用 `Start-Process` 明确运行 OpenSSH 并重定向三个标准流，避免 Windows PowerShell 5 的管道编码及异常 `PATHEXT` 导致启动、退出码不可靠。无需修改执行策略或 `PATHEXT`；如 `.ps1` 被阻止，由 Codex 工具用内联 `-Command` 执行同样内容。

```powershell
$SshAlias = 'labgpu'
$ControllerPython = '/srv/binder-env/bin/python'
$SkillCli = '/srv/trex-binder-design/scripts/trex_skill.py'
$Mailbox = '/srv/binder-campaign/mailbox'
$RequestId = '替换为本轮的32位request_id'
$LocalDecision = 'C:\Users\lab\AppData\Local\Temp\decision.json'
$RemoteDecisionDir = '/srv/binder-campaign/submitted-decisions'

if ($RequestId -notmatch '^[0-9a-f]{32}$') { throw 'Invalid request ID' }
$Tag = [guid]::NewGuid().ToString('N')
$RemoteDecision = "$RemoteDecisionDir/$RequestId-$Tag.json"
$TransportDir = Join-Path ([IO.Path]::GetTempPath()) "trex-transport-$Tag"
[IO.Directory]::CreateDirectory($TransportDir) | Out-Null
$EmptyInput = Join-Path $TransportDir 'empty.stdin'
[IO.File]::WriteAllBytes($EmptyInput, [byte[]]@())
$SshExe = Join-Path $env:WINDIR 'System32\OpenSSH\ssh.exe'
$ScpExe = Join-Path $env:WINDIR 'System32\OpenSSH\scp.exe'

function Invoke-Transport([string]$Exe, [string]$Arguments, [string]$Step) {
    $Out = Join-Path $TransportDir "$Step.stdout"
    $Err = Join-Path $TransportDir "$Step.stderr"
    $Process = Start-Process -FilePath $Exe -ArgumentList $Arguments `
        -NoNewWindow -Wait -PassThru -RedirectStandardInput $EmptyInput `
        -RedirectStandardOutput $Out -RedirectStandardError $Err
    if ($null -eq $Process.ExitCode) { throw "$Step returned no exit code" }
    $OutputText = [IO.File]::ReadAllText($Out)
    $ErrorText = [IO.File]::ReadAllText($Err)
    if ($Process.ExitCode -ne 0) {
        throw "$Step failed ($($Process.ExitCode)): $ErrorText $OutputText"
    }
    $OutputText
}

$Options = '-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=2'
Invoke-Transport $SshExe "-T $Options $SshAlias `"mkdir -p '$RemoteDecisionDir'`"" 'prepare'
Invoke-Transport $ScpExe "$Options `"$LocalDecision`" ${SshAlias}:$RemoteDecision" 'upload'
$Respond = "'$ControllerPython' '$SkillCli' respond --mailbox '$Mailbox' --request-id '$RequestId' --response '$RemoteDecision'"
$ReceiptText = Invoke-Transport $SshExe "-T $Options $SshAlias `"$Respond`"" 'respond'
$Receipt = $ReceiptText | ConvertFrom-Json
if ($Receipt.request_id -ne $RequestId -or $Receipt.status -ne 'accepted') {
    throw 'Unexpected response receipt; inspect the authoritative request before retrying'
}
$Receipt
```

使用前核对示例变量，不把用户提供的任意文本插入命令。`LocalDecision` 可含空格和中文，不能包含双引号。PowerShell 5 的 `Set-Content -Encoding UTF8` 会带 BOM，官方 `respond` 能接受；SCP 保留原始字节。也可用 `[IO.File]::WriteAllText(path, json, [Text.UTF8Encoding]::new($false))` 写无 BOM 文件。不要通过默认编码的 `Get-Content | ssh` 传递 JSON。

## macOS shell

下面在本地 shell 执行。`DECISION_FILE` 是已写好的 UTF-8 文件；路径及别名替换为实际值。远程暂存目录和文件名选择无空格的任务私有路径，文件名包含本轮 request ID 与新的唯一后缀。

```bash
SSH_ALIAS=labgpu
DECISION_FILE=/tmp/decision.json
REMOTE_DECISION=/srv/binder-campaign/submitted-decisions/REQUEST_ID-UNIQUE.json

ssh -T -o BatchMode=yes -o ConnectTimeout=10 "$SSH_ALIAS" \
  "mkdir -p '/srv/binder-campaign/submitted-decisions'" || exit "$?"
scp -o BatchMode=yes -o ConnectTimeout=10 "$DECISION_FILE" \
  "$SSH_ALIAS:$REMOTE_DECISION" || exit "$?"
ssh -T -o BatchMode=yes -o ConnectTimeout=10 "$SSH_ALIAS" \
  "/srv/binder-env/bin/python /srv/trex-binder-design/scripts/trex_skill.py respond --mailbox /srv/binder-campaign/mailbox --request-id REQUEST_ID --response '$REMOTE_DECISION'"
```

也可用已确认的非 TTY 管道 `ssh -T … '… respond --response -' < "$DECISION_FILE"`；该重定向语法属于 macOS shell，不直接粘到 PowerShell。

## 确认接收与消费

保留上传结果、官方回执和实际退出码。`accepted` 表示服务器接受提交；随后用官方 `show` 或 `requests --all` 核对同一 request/session 的 `consumed`，才表示控制器读到了回复。后续候选合法性、筛选和 GPU dispatch 仍须单独核实。

若连接中断而没有回执，先重新读取该 request 的权威状态：`accepted` 或 `consumed` 时不要再提交；仍为 `waiting` 且未过 deadline 时可原样重试同一份决策；`expired`、`cancelled` 或身份不一致时停止重试，按恢复流程处理。不要修改 request、deadline 或历史状态文件。无论采用文件还是 stdin，提交前都核对 deadline，完整 JSON 的上限为 1 MiB。
