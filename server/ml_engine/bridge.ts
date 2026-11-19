import { spawn, execFileSync, ChildProcess } from 'child_process';
import path from 'path';
import fs from 'fs';

interface PendingRequest {
  resolve: (data: any) => void;
  reject: (err: Error) => void;
  timer: NodeJS.Timeout;
}

export class PythonMlEngineBridge {
  private pyProcess: ChildProcess | null = null;
  private pythonExecutable: string | null = null;
  private stdoutBuffer = '';
  private queue: PendingRequest[] = [];
  private startupError: string | null = null;

  constructor() {
    this.resolveAndStart();
  }

  private resolvePythonExecutable(): string | null {
    const venvUnix = path.resolve('./.venv/bin/python');
    const venvWin = path.resolve('./.venv/Scripts/python.exe');
    const candidates = [
      process.env.OMNIBRAIN_PYTHON,
      fs.existsSync(venvUnix) ? venvUnix : null,
      fs.existsSync(venvWin) ? venvWin : null,
      'python3',
      'python',
      'py',
    ].filter((c): c is string => Boolean(c));

    for (const cmd of candidates) {
      try {
        const resolved = execFileSync(cmd, ['-c', 'import sys; print(sys.executable)'], {
          encoding: 'utf8',
          timeout: 3000,
          stdio: ['ignore', 'pipe', 'ignore'],
        }).trim();
        if (resolved) {
          return resolved;
        }
      } catch {
        // Probe next candidate
      }
    }
    return null;
  }

  private resolveAndStart() {
    const scriptPath = path.resolve('./server/ml_engine/engine.py');
    this.pythonExecutable = this.resolvePythonExecutable();

    if (!this.pythonExecutable) {
      this.startupError = 'No active Python interpreter found (checked OMNIBRAIN_PYTHON, python3, python, py).';
      console.error(this.startupError);
      return;
    }

    try {
      this.pyProcess = spawn(this.pythonExecutable, ['-u', scriptPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      this.pyProcess.on('error', (err) => {
        this.startupError = `Python engine spawn error: ${err.message}`;
        this.rejectAll(new Error(this.startupError));
      });

      this.pyProcess.on('exit', (code, signal) => {
        if (code !== 0 && code !== null) {
          const msg = `Python ML engine exited unexpectedly (code=${code}, signal=${signal}).`;
          this.startupError = msg;
          this.rejectAll(new Error(msg));
        }
        this.pyProcess = null;
      });

      this.pyProcess.stdout?.on('data', (chunk: Buffer) => {
        this.stdoutBuffer += chunk.toString('utf8');
        let newlineIdx = this.stdoutBuffer.indexOf('\n');
        while (newlineIdx !== -1) {
          const line = this.stdoutBuffer.slice(0, newlineIdx).trim();
          this.stdoutBuffer = this.stdoutBuffer.slice(newlineIdx + 1);
          if (line) {
            const pending = this.queue.shift();
            if (pending) {
              clearTimeout(pending.timer);
              try {
                pending.resolve(JSON.parse(line));
              } catch (err: any) {
                pending.reject(new Error(`Invalid JSON from Python engine: ${err.message}`));
              }
            }
          }
          newlineIdx = this.stdoutBuffer.indexOf('\n');
        }
      });

      this.pyProcess.stderr?.on('data', (err: Buffer) => {
        console.error('Python ML Engine stderr:', err.toString('utf8'));
      });
    } catch (err: any) {
      this.startupError = err.message;
    }
  }

  private rejectAll(err: Error) {
    while (this.queue.length > 0) {
      const item = this.queue.shift();
      if (item) {
        clearTimeout(item.timer);
        item.reject(err);
      }
    }
  }

  private sendCommand(payload: any, timeoutMs = 120000): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.pyProcess || !this.pyProcess.stdin) {
        this.resolveAndStart();
      }
      if (!this.pyProcess || !this.pyProcess.stdin) {
        return reject(new Error(this.startupError || 'Python ML engine process is not running.'));
      }

      const timer = setTimeout(() => {
        const idx = this.queue.findIndex((p) => p.resolve === resolve);
        if (idx !== -1) {
          this.queue.splice(idx, 1);
          reject(new Error(`Python ML engine timed out after ${timeoutMs}ms on action '${payload.action}'.`));
        }
      }, timeoutMs);

      this.queue.push({ resolve, reject, timer });

      const jsonStr = JSON.stringify(payload) + '\n';
      this.pyProcess.stdin.write(jsonStr, (err) => {
        if (err) {
          const idx = this.queue.findIndex((p) => p.resolve === resolve);
          if (idx !== -1) {
            this.queue.splice(idx, 1);
          }
          clearTimeout(timer);
          reject(err);
        }
      });
    });
  }

  getPythonExecutable(): string | null {
    return this.pythonExecutable;
  }

  async setBackend(backend: 'python_ref' | 'pytorch', device: 'auto' | 'cpu' | 'cuda' = 'auto') {
    return this.sendCommand({ action: 'backend', backend, device });
  }

  async runNumericalVerify() {
    return this.sendCommand({ action: 'numerical_verify' });
  }

  async setConfig(configName: string, seed = 42, backend?: string) {
    return this.sendCommand({ action: 'config', configName, seed, backend });
  }

  async getTelemetry() {
    return this.sendCommand({ action: 'telemetry' });
  }

  async getEstimates() {
    return this.sendCommand({ action: 'estimates' });
  }

  async trainPacked(options: {
    backend?: string;
    splitsPath?: string;
    epochs?: number;
    learningRate?: number;
    maxSteps?: number;
    evalInterval?: number;
    checkpointInterval?: number;
    resumeFrom?: string;
  } = {}) {
    const flagPath = path.resolve('./checkpoints/.cancel_training');
    if (fs.existsSync(flagPath)) {
      fs.unlinkSync(flagPath);
    }
    return this.sendCommand({ action: 'train_packed', ...options }, 300000);
  }

  async trainPhase7(options: {
    backend?: string;
    stage?: 'stageA' | 'stageB' | 'stageC' | 'custom';
    targetTokens?: number;
    batchSize?: number;
    learningRate?: number;
    minLr?: number;
    evalIntervalTokens?: number;
    resumeFrom?: string;
    checkpointBaseDir?: string;
    splitsPath?: string;
  } = {}) {
    const flagPath = path.resolve('./checkpoints/.cancel_training');
    if (fs.existsSync(flagPath)) {
      try {
        fs.unlinkSync(flagPath);
      } catch {
        // Ignore
      }
    }
    return this.sendCommand({ action: 'train_phase7', ...options }, 600000);
  }

  cancelTraining(): { requested: boolean } {
    const flagPath = path.resolve('./checkpoints/.cancel_training');
    fs.mkdirSync(path.dirname(flagPath), { recursive: true });
    fs.writeFileSync(flagPath, new Date().toISOString(), 'utf8');
    return { requested: true };
  }

  async inference(prompt: string, maxNewTokens = 32, temperature = 0.8) {
    return this.sendCommand({ action: 'inference', prompt, maxNewTokens, temperature });
  }

  async saveCheckpoint() {
    return this.sendCommand({ action: 'save' });
  }

  async loadCheckpoint(ckptPath: string) {
    return this.sendCommand({ action: 'load', path: ckptPath });
  }

  async loadPhase7Checkpoint(ckptPath: string) {
    return this.sendCommand({ action: 'load_phase7', path: ckptPath });
  }
}

export const globalPythonBridge = new PythonMlEngineBridge();
