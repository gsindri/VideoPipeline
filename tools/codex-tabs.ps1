# codex-tabs.ps1 — Open Codex sessions in Windows Terminal tabs
$ErrorActionPreference = 'Stop'

# Build the wt.exe argument string
$wtArgs = @(
    '-w', 'new',
    'new-tab',
        '--title', 'Trador - Codex',
        '--tabColor', '#7e57c2',
        '--suppressApplicationTitle',
        'wsl.exe', '-d', 'Ubuntu', '--', 'bash', '-lc', '~/bin/codexw',
    ';',
    'new-tab',
        '--title', 'VP - Codex',
        '--tabColor', '#26a69a',
        '--suppressApplicationTitle',
        'wsl.exe', '-d', 'Ubuntu', '--', 'bash', '-lc', '~/bin/codex_videopipeline',
    ';',
    'new-tab',
        '--title', 'VP - Codex 2',
        '--tabColor', '#26a69a',
        '--suppressApplicationTitle',
        'wsl.exe', '-d', 'Ubuntu', '--', 'bash', '-lc', '~/bin/codex_videopipeline'
)

Start-Process wt.exe -ArgumentList $wtArgs
