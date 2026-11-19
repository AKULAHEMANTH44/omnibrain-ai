import React, { useState } from 'react';
import { Database, FileText, Upload, Search, Trash2, ShieldCheck, Sparkles } from 'lucide-react';

export const KnowledgeLibraryWorkspace: React.FC = () => {
  const [documents, setDocuments] = useState([
    { id: '1', title: 'OMNIBRAIN Technical Whitepaper v2.0.pdf', size: '4.2 MB', vectors: '14,250 tokens', date: 'Nov 18, 2026' },
    { id: '2', title: 'Transformer Model Scaling Laws & Attention Optimization.pdf', size: '2.8 MB', vectors: '9,840 tokens', date: 'Nov 15, 2026' },
    { id: '3', title: 'Quantum Cryptography Protocol Specification.docx', size: '1.5 MB', vectors: '5,120 tokens', date: 'Nov 12, 2026' },
  ]);

  const handleUpload = () => {
    const title = prompt('Enter document name to add to Knowledge Library:');
    if (!title) return;
    const newDoc = {
      id: Date.now().toString(),
      title: title.endsWith('.pdf') || title.endsWith('.docx') ? title : title + '.pdf',
      size: `${(Math.random() * 3 + 1).toFixed(1)} MB`,
      vectors: `${Math.floor(Math.random() * 10000 + 2000)} tokens`,
      date: 'Today',
    };
    setDocuments([newDoc, ...documents]);
  };

  const handleDelete = (id: string) => {
    setDocuments(documents.filter(d => d.id !== id));
  };

  return (
    <div className="flex-1 flex flex-col h-screen bg-[#07080a] overflow-y-auto p-6 md:p-10">
      <div className="max-w-6xl mx-auto w-full space-y-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-6">
          <div>
            <div className="flex items-center gap-2 text-amber-400 text-xs font-mono uppercase tracking-widest mb-1">
              <Database className="w-4 h-4" />
              <span>OMNIBRAIN Knowledge Library</span>
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">
              Vector Memory Vault & Document Index
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Manage your uploaded documents, research papers, and vector embeddings for semantic RAG retrieval.
            </p>
          </div>

          <button
            onClick={handleUpload}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-black text-xs font-bold transition-all shadow-[0_0_20px_rgba(245,158,11,0.3)] cursor-pointer"
          >
            <Upload className="w-4 h-4 fill-black" />
            <span>Upload Document</span>
          </button>
        </div>

        {/* Documents List */}
        <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl overflow-hidden shadow-xl">
          <div className="p-4 bg-[#0a0c10] border-b border-white/10 flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>DOCUMENT TITLE</span>
            <div className="flex items-center gap-12">
              <span>SIZE / TOKENS</span>
              <span>INDEXED DATE</span>
              <span>ACTION</span>
            </div>
          </div>

          <div className="divide-y divide-white/5">
            {documents.length === 0 ? (
              <div className="p-8 text-center text-xs text-slate-500">
                No documents in the knowledge library. Click "Upload Document" to add files.
              </div>
            ) : (
              documents.map((doc) => (
                <div key={doc.id} className="p-4 flex items-center justify-between hover:bg-white/5 transition-colors">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
                      <FileText className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-semibold text-white">{doc.title}</h3>
                      <span className="text-[10px] text-emerald-400 font-mono flex items-center gap-1 mt-0.5">
                        <ShieldCheck className="w-3 h-3" /> Vector Indexed
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-12 text-xs text-slate-300 font-mono">
                    <span>{doc.size} ({doc.vectors})</span>
                    <span>{doc.date}</span>
                    <button
                      onClick={() => handleDelete(doc.id)}
                      className="p-2 hover:text-rose-400 transition-colors"
                      title="Delete document"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
