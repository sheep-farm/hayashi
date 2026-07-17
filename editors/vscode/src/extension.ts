import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

let outputChannel: vscode.OutputChannel;
let debugChannel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
    outputChannel = vscode.window.createOutputChannel('Hayashi');
    context.subscriptions.push(outputChannel);
    debugChannel = vscode.window.createOutputChannel('Hayashi Debug');
    context.subscriptions.push(debugChannel);

    // Formatter
    context.subscriptions.push(
        vscode.languages.registerDocumentFormattingEditProvider('hayashi', {
            provideDocumentFormattingEdits(
                document: vscode.TextDocument,
                _options: vscode.FormattingOptions,
                _token: vscode.CancellationToken
            ): vscode.TextEdit[] {
                const cfg = vscode.workspace.getConfiguration('hayashi.format');
                const indentSize = cfg.get<number>('indentSize', 2);
                const alignEquals = cfg.get<boolean>('alignEquals', true);
                const indent = ' '.repeat(indentSize);

                const lines = document.getText().split(/\r?\n/);
                const formatted: string[] = [];
                let depth = 0;
                const assignBlocks: number[][] = [];
                let blockStart = -1;

                for (let i = 0; i < lines.length; i++) {
                    let line = lines[i].trim();

                    if (/^[}\]]/.test(line) || /^end\b/.test(line) || /^else\b/.test(line) || /^catch\b/.test(line)) {
                        depth = Math.max(0, depth - 1);
                    }

                    const isAssign = /^\w+\s*=/.test(line) && !/^(if|for|while|fn|match|let|const)\b/.test(line);
                    if (isAssign) {
                        if (blockStart === -1) blockStart = i;
                    } else {
                        if (blockStart !== -1 && i - blockStart >= 2) {
                            assignBlocks.push([blockStart, i - 1]);
                        }
                        blockStart = -1;
                    }

                    if (line.length > 0) {
                        line = indent.repeat(depth) + line;
                    }
                    formatted.push(line);

                    if (/[{[]\s*$/.test(line) || (/\b(fn|for|while|if|else|input|match|try)\b.*$/.test(line) && !/;\s*$/.test(line))) {
                        if (/[{[]\s*$/.test(line) || /\b(fn|input|match|try)\b/.test(line)) {
                            depth++;
                        } else if (/\b(if|else|for|while)\b/.test(line) && !/[{};]\s*$/.test(line)) {
                            depth++;
                        }
                    }
                }

                if (blockStart !== -1 && lines.length - blockStart >= 2) {
                    assignBlocks.push([blockStart, lines.length - 1]);
                }

                if (alignEquals) {
                    for (const [start, end] of assignBlocks) {
                        let maxLen = 0;
                        for (let i = start; i <= end; i++) {
                            const m = formatted[i].match(/^(\s*\w+\s*)(=)/);
                            if (m) {
                                maxLen = Math.max(maxLen, m[1].trimEnd().length);
                            }
                        }
                        for (let i = start; i <= end; i++) {
                            const m = formatted[i].match(/^(\s*)(\w+)(\s*)(=)(\s.*)$/);
                            if (m) {
                                const leading = m[1];
                                const name = m[2];
                                const pad = ' '.repeat(maxLen - name.length);
                                formatted[i] = leading + name + pad + ' = ' + m[5].trimStart();
                            }
                        }
                    }
                }

                const edits: vscode.TextEdit[] = [];
                const fullRange = new vscode.Range(
                    document.positionAt(0),
                    document.positionAt(document.getText().length)
                );
                edits.push(vscode.TextEdit.replace(fullRange, formatted.join('\n')));
                return edits;
            },
        })
    );

    // Runner: Run File
    context.subscriptions.push(
        vscode.commands.registerCommand('hayashi.runFile', () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'hayashi') {
                vscode.window.showWarningMessage('No Hayashi file open');
                return;
            }
            runHayashi(editor.document.uri, editor.document.getText());
        })
    );

    // Runner: Run Selection
    context.subscriptions.push(
        vscode.commands.registerCommand('hayashi.runSelection', () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'hayashi') {
                vscode.window.showWarningMessage('No Hayashi file open');
                return;
            }
            const selection = editor.selection;
            if (selection.isEmpty) {
                vscode.window.showInformationMessage('No selection. Use Run File instead.');
                return;
            }
            const text = editor.document.getText(selection);
            runHayashi(editor.document.uri, text);
        })
    );

    // Debugger
    context.subscriptions.push(
        vscode.debug.registerDebugAdapterDescriptorFactory('hayashi', new HayashiDebugAdapterFactory())
    );
    context.subscriptions.push(
        vscode.debug.registerDebugConfigurationProvider('hayashi', new HayashiConfigurationProvider())
    );
}

function runHayashi(uri: vscode.Uri, code: string) {
    const cfg = vscode.workspace.getConfiguration('hayashi.runner');
    const executable = cfg.get<string>('executable', 'hay');
    const clearOutput = cfg.get<boolean>('clearOutput', true);

    if (clearOutput) {
        outputChannel.clear();
    }

    const fileName = path.basename(uri.fsPath);
    outputChannel.appendLine(`$ hay ${fileName}`);
    outputChannel.show(true);

    const tmpDir = path.join(os.tmpdir(), 'vscode-hayashi');
    if (!fs.existsSync(tmpDir)) {
        fs.mkdirSync(tmpDir, { recursive: true });
    }
    const tmpFile = path.join(tmpDir, `run_${Date.now()}.hay`);
    fs.writeFileSync(tmpFile, code);

    const proc = cp.spawn(executable, [tmpFile], {
        cwd: path.dirname(uri.fsPath),
        env: { ...process.env },
    });

    proc.stdout.on('data', (data: Buffer) => {
        outputChannel.append(data.toString());
    });

    proc.stderr.on('data', (data: Buffer) => {
        outputChannel.append(data.toString());
    });

    proc.on('close', (code: number) => {
        if (code !== 0) {
            outputChannel.appendLine(`\n[exited with code ${code}]`);
        }
        try { fs.unlinkSync(tmpFile); } catch {}
    });

    proc.on('error', (err: Error) => {
        outputChannel.appendLine(`Error: ${err.message}`);
        outputChannel.appendLine(
            "Make sure 'hay' is installed and in your PATH. " +
            'Configure with: hayashi.runner.executable in settings.'
        );
    });
}

class HayashiDebugAdapterFactory implements vscode.DebugAdapterDescriptorFactory {
    createDebugAdapterDescriptor(
        session: vscode.DebugSession,
        _executable: vscode.DebugAdapterExecutable | undefined
    ): vscode.ProviderResult<vscode.DebugAdapterDescriptor> {
        const config = session.configuration;
        const executable = String(config.runtimeExecutable || 'hay');
        const args: string[] = Array.isArray(config.runtimeArgs)
            ? config.runtimeArgs.map(String)
            : ['dap'];

        let program = config.program ? String(config.program) : undefined;
        if (!program || program === '${file}') {
            for (const editor of vscode.window.visibleTextEditors) {
                if (editor.document.languageId === 'hayashi') {
                    program = editor.document.uri.fsPath;
                    break;
                }
            }
        }
        if (!program || program.includes('extension-output')) {
            vscode.window.showErrorMessage('No Hayashi file selected for debugging');
            throw new Error('No Hayashi file selected for debugging');
        }

        args.push(program);
        return new vscode.DebugAdapterExecutable(executable, args);
    }
}

class HayashiConfigurationProvider implements vscode.DebugConfigurationProvider {
    resolveDebugConfiguration(
        folder: vscode.WorkspaceFolder | undefined,
        config: vscode.DebugConfiguration
    ): vscode.ProviderResult<vscode.DebugConfiguration> {
        if (!config.program || config.program === '${file}') {
            const editor = vscode.window.activeTextEditor;
            if (editor) {
                config.program = editor.document.uri.fsPath;
            }
        }
        if (typeof config.program === 'string' && config.program.includes('${workspaceFolder}')) {
            const wf = folder?.uri.fsPath || vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
            if (wf) {
                config.program = config.program.replace('${workspaceFolder}', wf);
            }
        }
        if (!config.program) {
            vscode.window.showErrorMessage('No Hayashi file selected for debugging');
            return undefined;
        }
        if (!config.runtimeExecutable) {
            config.runtimeExecutable = 'hay';
        }
        if (!Array.isArray(config.runtimeArgs) || config.runtimeArgs.length === 0) {
            config.runtimeArgs = ['dap'];
        }
        if (config.type === 'hayashi' && !config.request) {
            config.request = 'launch';
        }
        return config;
    }
}

export function deactivate() {}
