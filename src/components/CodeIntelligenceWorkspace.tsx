import React, { useState } from 'react';
import { Code2, Play, Copy, Check, Terminal, Cpu, Sparkles } from 'lucide-react';

export const CodeIntelligenceWorkspace: React.FC = () => {
  const [language, setLanguage] = useState('python');
  const [code, setCode] = useState(`# OMNIBRAIN Neural Transformer Attention Block
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)`);

  const [output, setOutput] = useState('Ready for sandbox execution. Click "Run Code Analysis" to simulate neural execution and static analysis.');
  const [isExecuting, setIsExecuting] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleRun = () => {
    setIsExecuting(true);
    setTimeout(() => {
      setOutput(`[OMNIBRAIN Code Sandbox v2.5]
Status: SUCCESS (0 errors, 0 warnings)
Language: ${language.toUpperCase()}
Execution Time: 14.2ms
Memory Peak: 42.8 MB
AST Validation: Passed
Type Check: Strict TypeScript / Python Mypy Clean
Output Log:
> Tensor shape validated: torch.Size([32, 128, 4096])
> Multi-head attention forward pass completed successfully.
> Gradient norm: 0.8412`);
      setIsExecuting(false);
    }, 800);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex-1 flex flex-col h-screen bg-[#07080a] overflow-y-auto p-6 md:p-10">
      <div className="max-w-6xl mx-auto w-full space-y-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-6">
          <div>
            <div className="flex items-center gap-2 text-blue-400 text-xs font-mono uppercase tracking-widest mb-1">
              <Code2 className="w-4 h-4" />
              <span>OMNIBRAIN Code Intelligence</span>
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">
              Advanced Code Sandbox & Neural Analyzer
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Analyze, optimize, and simulate code execution with OMNIBRAIN AI code intelligence.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="bg-[#12151c] border border-white/10 rounded-xl px-4 py-2 text-xs text-white focus:outline-none focus:border-blue-500"
            >
              <option value="python">Python 3.11</option>
              <option value="typescript">TypeScript</option>
              <option value="rust">Rust</option>
              <option value="cpp">C++ 20</option>
            </select>

            <button
              onClick={handleRun}
              disabled={isExecuting}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-500 hover:bg-blue-400 text-black text-xs font-bold transition-all shadow-[0_0_20px_rgba(59,130,246,0.3)]"
            >
              <Play className="w-4 h-4 fill-black" />
              <span>{isExecuting ? 'Running...' : 'Run Analysis'}</span>
            </button>
          </div>
        </div>

        {/* Editor & Output Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Code Editor Box */}
          <div className="lg:col-span-2 bg-[#12151c] border border-white/10 rounded-2xl overflow-hidden shadow-xl flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 bg-[#0a0c10] border-b border-white/10">
              <span className="text-xs font-mono text-slate-400">transformer_block.{language === 'python' ? 'py' : 'ts'}</span>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy Code'}</span>
              </button>
            </div>
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              rows={16}
              className="w-full bg-[#12151c] text-cyan-300 font-mono text-xs p-5 focus:outline-none resize-none leading-relaxed"
            />
          </div>

          {/* Execution Output Box */}
          <div className="bg-[#12151c] border border-white/10 rounded-2xl overflow-hidden shadow-xl flex flex-col">
            <div className="flex items-center gap-2 px-4 py-3 bg-[#0a0c10] border-b border-white/10">
              <Terminal className="w-4 h-4 text-emerald-400" />
              <span className="text-xs font-mono text-slate-300">Sandbox Output</span>
            </div>
            <div className="p-5 font-mono text-xs text-slate-300 bg-[#07080a] flex-1 overflow-y-auto whitespace-pre-wrap leading-relaxed">
              {output}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
