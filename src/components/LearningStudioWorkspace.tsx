import React, { useState } from 'react';
import { GraduationCap, BookOpen, CheckCircle, Play, Sparkles, Clock, Award } from 'lucide-react';

export const LearningStudioWorkspace: React.FC = () => {
  const [selectedCourse, setSelectedCourse] = useState<number | null>(null);

  const courses = [
    {
      id: 1,
      title: 'Advanced Transformer Architectures & Attention Mechanisms',
      category: 'Machine Learning',
      level: 'Expert',
      duration: '4 hours',
      modules: 8,
      progress: 75,
      description: 'Master self-attention, rotary position embeddings (RoPE), FlashAttention, and mixture-of-experts (MoE) scaling.'
    },
    {
      id: 2,
      title: 'Quantum Computing & Algorithms for Cryptography',
      category: 'Quantum Science',
      level: 'Advanced',
      duration: '6 hours',
      modules: 10,
      progress: 30,
      description: 'Explore qubits, quantum gates, Shor s algorithm, and quantum error correction principles.'
    },
    {
      id: 3,
      title: 'Distributed Systems & High-Throughput RAG Pipelines',
      category: 'Systems Engineering',
      level: 'Intermediate',
      duration: '5 hours',
      modules: 6,
      progress: 0,
      description: 'Build enterprise retrieval-augmented generation systems with vector databases and semantic caching.'
    },
    {
      id: 4,
      title: 'Neurosymbolic AI & Automated Reasoning Systems',
      category: 'Artificial Intelligence',
      level: 'Advanced',
      duration: '4.5 hours',
      modules: 7,
      progress: 0,
      description: 'Combine neural network representation learning with symbolic logic theorem provers.'
    },
  ];

  return (
    <div className="flex-1 flex flex-col h-screen bg-[#07080a] overflow-y-auto p-6 md:p-10">
      <div className="max-w-6xl mx-auto w-full space-y-8">
        {/* Header */}
        <div className="border-b border-white/10 pb-6">
          <div className="flex items-center gap-2 text-emerald-400 text-xs font-mono uppercase tracking-widest mb-1">
            <GraduationCap className="w-4 h-4" />
            <span>OMNIBRAIN Learning Studio</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">
            AI-Powered Curricula & Advanced Mastery
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Personalized expert learning paths across mathematics, computer science, physics, and engineering.
          </p>
        </div>

        {/* Courses Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {courses.map((course) => (
            <div
              key={course.id}
              className="bg-[#12151c]/90 border border-white/10 hover:border-emerald-500/40 rounded-2xl p-6 shadow-xl flex flex-col justify-between transition-all group"
            >
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono uppercase tracking-wider text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-lg border border-emerald-500/20">
                    {course.category}
                  </span>
                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <Clock className="w-3.5 h-3.5" />
                    <span>{course.duration}</span>
                  </div>
                </div>

                <h3 className="text-lg font-bold text-white group-hover:text-emerald-300 transition-colors">
                  {course.title}
                </h3>

                <p className="text-xs text-slate-400 leading-relaxed">
                  {course.description}
                </p>
              </div>

              <div className="pt-6 mt-6 border-t border-white/10 space-y-4">
                {course.progress > 0 && (
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs text-slate-400">
                      <span>Progress</span>
                      <span className="text-emerald-400 font-mono font-bold">{course.progress}%</span>
                    </div>
                    <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${course.progress}%` }}></div>
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400">{course.modules} Modules · {course.level}</span>
                  <button
                    onClick={() => alert(`Starting interactive learning module: "${course.title}"`)}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-black text-xs font-bold transition-all shadow-[0_0_15px_rgba(16,185,129,0.2)]"
                  >
                    <Play className="w-3.5 h-3.5 fill-black" />
                    <span>{course.progress > 0 ? 'Resume Course' : 'Start Learning'}</span>
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
