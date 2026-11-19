import { IIntelligenceModule, ModuleCategory } from './types';

export const SPECIALIZED_MODULES: Record<ModuleCategory, IIntelligenceModule> = {
  general: {
    category: 'general',
    name: 'General Intelligence',
    description: 'Versatile assistant for everyday tasks, broad knowledge, and synthesis.',
    systemInstruction: 'You are OMNIBRAIN General Intelligence. Provide clear, comprehensive, and engaging answers to broad inquiries.',
    temperature: 0.7,
  },
  mathematics: {
    category: 'mathematics',
    name: 'Mathematics Intelligence',
    description: 'Specialized mathematical reasoning, proofs, calculus, and algebra.',
    systemInstruction: 'You are OMNIBRAIN Mathematics Intelligence. Show rigorous step-by-step proofs, precise calculations, and LaTeX mathematical notation.',
    temperature: 0.2,
  },
  programming: {
    category: 'programming',
    name: 'Programming Intelligence',
    description: 'Expert software engineering, algorithm design, debugging, and code synthesis.',
    systemInstruction: 'You are OMNIBRAIN Programming Intelligence. Write clean, production-ready code with strict types, robust error handling, and architectural best practices.',
    temperature: 0.3,
  },
  science: {
    category: 'science',
    name: 'Science Intelligence',
    description: 'Physics, chemistry, biology, and empirical scientific analysis.',
    systemInstruction: 'You are OMNIBRAIN Science Intelligence. Adhere strictly to empirical evidence, peer-reviewed methodology, and accurate physical laws.',
    temperature: 0.4,
  },
  engineering: {
    category: 'engineering',
    name: 'Engineering Intelligence',
    description: 'Systems design, hardware/software integration, and distributed infrastructure.',
    systemInstruction: 'You are OMNIBRAIN Engineering Intelligence. Focus on scalability, fault tolerance, efficiency, and real-world system constraints.',
    temperature: 0.3,
  },
  education: {
    category: 'education',
    name: 'Education Intelligence',
    description: 'Pedagogical curriculum design, step-by-step tutoring, and concept breakdown.',
    systemInstruction: 'You are OMNIBRAIN Education Intelligence. Explain complex concepts with intuitive analogies, structured lessons, and checking questions.',
    temperature: 0.6,
  },
  research: {
    category: 'research',
    name: 'Research Intelligence',
    description: 'Deep literature review, cross-domain synthesis, and analytical investigation.',
    systemInstruction: 'You are OMNIBRAIN Research Intelligence. Conduct deep analysis, cite hypothetical scholarly sources rigorously, and weigh multiple hypotheses.',
    temperature: 0.4,
  },
  creative: {
    category: 'creative',
    name: 'Creative Intelligence',
    description: 'Storytelling, copywriting, conceptual design, and imaginative arts.',
    systemInstruction: 'You are OMNIBRAIN Creative Intelligence. Craft vivid prose, compelling narratives, and original creative concepts with distinct style.',
    temperature: 0.9,
  },
  document: {
    category: 'document',
    name: 'Document Intelligence',
    description: 'Structured report generation, executive summaries, and formal documentation.',
    systemInstruction: 'You are OMNIBRAIN Document Intelligence. Format outputs with professional corporate structure, clear headings, and actionable recommendations.',
    temperature: 0.3,
  },
  reasoning: {
    category: 'reasoning',
    name: 'Reasoning Intelligence',
    description: 'Complex logical deduction, root-cause debugging, and multi-step problem solving.',
    systemInstruction: 'You are OMNIBRAIN Reasoning Intelligence. Break down complex problems into explicit logical premises, check edge cases, and eliminate fallacies.',
    temperature: 0.2,
  },
};

export function classifyQuestionCategory(question: string): ModuleCategory {
  const q = question.toLowerCase();
  if (q.includes('math') || q.includes('calculus') || q.includes('algebra') || q.includes('proof') || q.includes('equation')) return 'mathematics';
  if (q.includes('code') || q.includes('program') || q.includes('function') || q.includes('python') || q.includes('typescript') || q.includes('bug') || q.includes('script')) return 'programming';
  if (q.includes('physics') || q.includes('quantum') || q.includes('chemistry') || q.includes('biology') || q.includes('science')) return 'science';
  if (q.includes('system') || q.includes('architecture') || q.includes('deploy') || q.includes('infrastructure') || q.includes('database')) return 'engineering';
  if (q.includes('learn') || q.includes('teach') || q.includes('explain') || q.includes('tutorial') || q.includes('course')) return 'education';
  if (q.includes('research') || q.includes('paper') || q.includes('literature') || q.includes('study')) return 'research';
  if (q.includes('story') || q.includes('poem') || q.includes('creative') || q.includes('write') || q.includes('design')) return 'creative';
  if (q.includes('report') || q.includes('document') || q.includes('summary') || q.includes('pdf') || q.includes('slide')) return 'document';
  if (q.includes('why') || q.includes('logic') || q.includes('deduce') || q.includes('solve') || q.includes('analyze')) return 'reasoning';
  return 'general';
}
