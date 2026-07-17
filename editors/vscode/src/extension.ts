import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    context.subscriptions.push(
        vscode.debug.registerDebugAdapterDescriptorFactory('hayashi', new HayashiDebugAdapterFactory())
    );
    context.subscriptions.push(
        vscode.debug.registerDebugConfigurationProvider('hayashi', new HayashiConfigurationProvider())
    );
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
        return new vscode.DebugAdapterExecutable(executable, args);
    }
}

class HayashiConfigurationProvider implements vscode.DebugConfigurationProvider {
    resolveDebugConfiguration(
        _folder: vscode.WorkspaceFolder | undefined,
        config: vscode.DebugConfiguration
    ): vscode.ProviderResult<vscode.DebugConfiguration> {
        if (!config.program) {
            config.program = '${file}';
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
