import React, { useState } from 'react';
import {
  FileText,
  Presentation,
  FileCode,
  Network,
  Workflow,
  BookOpen,
  HelpCircle,
  Briefcase,
  Sparkles,
  Download,
  Copy,
  Check,
  Loader2
} from 'lucide-react';
import { StudioDocument } from '../types';

export const CreateStudioWorkspace: React.FC = () => {
  const [selectedType, setSelectedType] = useState<StudioDocument['type']>('report');
  const [topic, setTopic] = useState('');
  const [depth, setDepth] = useState('detailed');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentDoc, setCurrentDoc] = useState<StudioDocument | null>(null);
  const [copied, setCopied] = useState(false);

  const documentTypes = [
    { id: 'report', label: 'Project Report', icon: Briefcase, desc: 'Comprehensive executive reports and strategic analyses' },
    { id: 'pdf', label: 'PDF Summary', icon: FileText, desc: 'Formatted multi-section PDF brief with citations' },
    { id: 'powerpoint', label: 'PowerPoint Slides', icon: Presentation, desc: 'Slide-by-slide executive presentation outlines' },
    { id: 'word', label: 'Word Document', icon: FileText, desc: 'Detailed academic or corporate document structure' },
    { id: 'diagram', label: 'System Diagram', icon: Network, desc: 'Architecture & component diagram specification (Mermaid)' },
    { id: 'flowchart', label: 'Process Flowchart', icon: Workflow, desc: 'Step-by-step algorithmic or business workflow' },
    { id: 'notes', label: 'Study Notes', icon: BookOpen, desc: 'Structured revision notes with key takeaways' },
    { id: 'quiz', label: 'Question Paper', icon: HelpCircle, desc: 'Rigorous exam or assessment questions with answers' },
  ];

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!topic.trim() || isGenerating) return;

    setIsGenerating(true);
    try {
      const res = await fetch('/api/create-studio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: selectedType,
          topic: topic.trim(),
          format: selectedType.toUpperCase(),
          depth,
        }),
      });

      const data = await res.json();
      if (res.ok) {
        setCurrentDoc({
          id: Date.now().toString(),
          title: topic.trim(),
          type: selectedType,
          content: data.content,
          createdAt: new Date().toLocaleTimeString(),
          topic: topic.trim(),
        });
      } else {
        alert(data.error || 'Failed to generate studio document.');
      }
    } catch (err: any) {
      alert('Network error: ' + err.message);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCopy = () => {
    if (!currentDoc) return;
    navigator.clipboard.writeText(currentDoc.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    if (!currentDoc) return;
    const blob = new Blob([currentDoc.content], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentDoc.type}-${currentDoc.topic.replace(/\s+/g, '_')}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex-1 flex flex-col h-screen bg-[#07080a] overflow-y-auto p-6 md:p-10">
      <div className="max-w-6xl mx-auto w-full space-y-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-6">
          <div>
            <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono uppercase tracking-widest mb-1">
              <Sparkles className="w-4 h-4" />
              <span>OMNIBRAIN Create Studio</span>
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">
              Professional Document & Diagram Generation
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Generate publication-grade reports, slide outlines, study notes, and system diagrams instantly.
            </p>
          </div>
        </div>

        {/* Document Type Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {documentTypes.map((item) => {
            const Icon = item.icon;
            const isSelected = selectedType === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setSelectedType(item.id as any)}
                className={`p-4 rounded-2xl border text-left transition-all flex flex-col justify-between ${
                  isSelected
                    ? 'bg-cyan-500/10 border-cyan-500 text-white shadow-[0_0_20px_rgba(0,240,255,0.15)]'
                    : 'bg-[#12151c]/80 border-white/10 text-slate-300 hover:bg-[#1a1f2c] hover:border-white/20'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className={`w-9 h-9 rounded-xl flex items-center justify-center ${
                    isSelected ? 'bg-cyan-500 text-black' : 'bg-white/5 text-cyan-400'
                  }`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  {isSelected && <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>}
                </div>
                <div>
                  <h3 className="font-bold text-sm text-white mb-1">{item.label}</h3>
                  <p className="text-[11px] text-slate-400 line-clamp-2">{item.desc}</p>
                </div>
              </button>
            );
          })}
        </div>

        {/* Generation Form */}
        <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-6 shadow-xl space-y-5">
          <form onSubmit={handleGenerate} className="space-y-4">
            <div className="space-y-2">
              <label className="text-xs font-bold uppercase tracking-wider text-slate-300">
                Topic / Subject / Project Name
              </label>
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g., Quantum Computing Breakthroughs 2026, Enterprise Microservices Migration, Neural Network Optimizations..."
                className="w-full bg-[#0a0c10] border border-white/10 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500/60"
              />
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
              <div className="flex items-center gap-3 w-full sm:w-auto">
                <span className="text-xs text-slate-400 font-medium">Depth Level:</span>
                <select
                  value={depth}
                  onChange={(e) => setDepth(e.target.value)}
                  className="bg-[#0a0c10] border border-white/10 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500/50"
                >
                  <option value="concise">Concise Summary</option>
                  <option value="detailed">Detailed Professional</option>
                  <option value="comprehensive">Comprehensive Academic</option>
                </select>
              </div>

              <button
                type="submit"
                disabled={!topic.trim() || isGenerating}
                className={`w-full sm:w-auto px-6 py-3 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all ${
                  topic.trim() && !isGenerating
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-black hover:scale-105 shadow-[0_0_25px_rgba(0,240,255,0.3)] cursor-pointer'
                    : 'bg-white/5 text-slate-600 cursor-not-allowed'
                }`}
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin text-black" />
                    <span>Generating Studio Document...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 text-black" />
                    <span>Generate Document</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Preview Section */}
        {currentDoc && (
          <div className="bg-[#12151c]/90 border border-cyan-500/30 rounded-2xl p-6 shadow-2xl space-y-4 animate-fade-in">
            <div className="flex items-center justify-between border-b border-white/10 pb-4">
              <div>
                <span className="text-[10px] font-mono uppercase tracking-widest text-cyan-400 px-2 py-0.5 rounded bg-cyan-500/10">
                  {currentDoc.type.toUpperCase()} Generated
                </span>
                <h2 className="text-xl font-bold text-white mt-1">{currentDoc.title}</h2>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-slate-200 text-xs font-medium transition-colors"
                >
                  {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black text-xs font-bold transition-all shadow-[0_0_15px_rgba(0,240,255,0.2)]"
                >
                  <Download className="w-4 h-4" />
                  <span>Download Markdown</span>
                </button>
              </div>
            </div>

            <div className="bg-[#07080a] border border-white/10 rounded-xl p-6 text-slate-200 text-sm leading-relaxed whitespace-pre-wrap font-['Plus_Jakarta_Sans',sans-serif] max-h-[500px] overflow-y-auto">
              {currentDoc.content}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
