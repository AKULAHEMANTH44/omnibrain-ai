import React, { useState } from 'react';
import { NavigationTab, Conversation, Message } from './types';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { ChatWorkspace } from './components/ChatWorkspace';
import { CreateStudioWorkspace } from './components/CreateStudioWorkspace';
import { LearningStudioWorkspace } from './components/LearningStudioWorkspace';
import { CodeIntelligenceWorkspace } from './components/CodeIntelligenceWorkspace';
import { KnowledgeLibraryWorkspace } from './components/KnowledgeLibraryWorkspace';
import { MlArchitectureWorkspace } from './components/MlArchitectureWorkspace';
import { SettingsModal } from './components/SettingsModal';

export default function App() {
  const [currentTab, setTab] = useState<NavigationTab>('chat');
  const [selectedModel, setSelectedModel] = useState('gemini-4-hydrogen');
  const [thinkingLevel, setThinkingLevel] = useState('MEDIUM');

  const [conversations, setConversations] = useState<Conversation[]>([
    {
      id: '1',
      title: 'Neural Architecture Design & MoE Scaling',
      messages: [
        {
          id: 'm1',
          role: 'user',
          content: 'Design a distributed Transformer LLM architecture with multi-head attention and pipeline parallelism.',
          timestamp: '10:45 AM',
        },
        {
          id: 'm2',
          role: 'assistant',
          content: `Here is an elite design blueprint for a distributed Transformer LLM architecture featuring Mixture-of-Experts (MoE) routing and pipeline parallelism:

### 1. Model Topology & Dimensions
- **Hidden Dimension ($d_{\text{model}}$)**: 4,096
- **Number of Layers**: 32 Transformer blocks
- **Attention Heads**: 32 query heads, 8 key-value heads (Grouped-Query Attention)
- **Context Window**: 128,000 tokens with Rotary Position Embeddings (RoPE)

### 2. Mixture-of-Experts (MoE) Routing
- **Total Experts**: 8 expert FFN networks per layer
- **Top-$k$ Routing**: $k = 2$ active experts per token
- **Router Gate**: Softmax gating with load balancing auxiliary loss to prevent expert starvation.

### 3. Pipeline & Tensor Parallelism
- **Megatron-LM Tensor Parallelism**: Splits attention projections and MLP layers across 8 GPUs per node.
- **ZeRO Stage 3 Optimizer Partitioning**: Distributes optimizer states, gradients, and model parameters across data-parallel ranks.`,
          timestamp: '10:45 AM',
          modelUsed: 'gemini-4-hydrogen',
        }
      ],
      updatedAt: '10:45 AM',
      model: 'gemini-4-hydrogen',
    }
  ]);

  const [currentConversationId, setCurrentConversationId] = useState<string>('1');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const currentConversation = conversations.find(c => c.id === currentConversationId) || conversations[0];

  const newChat = () => {
    const newId = Date.now().toString();
    const newConv: Conversation = {
      id: newId,
      title: 'New Neural Chat',
      messages: [],
      updatedAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      model: selectedModel,
    };
    setConversations([newConv, ...conversations]);
    setCurrentConversationId(newId);
    setTab('chat');
  };

  const selectConversation = (id: string) => {
    setCurrentConversationId(id);
  };

  const deleteConversation = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = conversations.filter(c => c.id !== id);
    setConversations(updated);
    if (currentConversationId === id && updated.length > 0) {
      setCurrentConversationId(updated[0].id);
    }
  };

  const handleSendMessage = async (content: string) => {
    setError(null);
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    const updatedMessages = [...(currentConversation?.messages || []), userMessage];

    // Update conversation title if first message
    const title = currentConversation?.messages.length === 0
      ? (content.length > 30 ? content.substring(0, 30) + '...' : content)
      : currentConversation?.title || 'Neural Chat';

    const updatedConversations = conversations.map(c =>
      c.id === currentConversationId
        ? { ...c, title, messages: updatedMessages, updatedAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }
        : c
    );

    setConversations(updatedConversations);
    setIsLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: updatedMessages.map(m => ({ role: m.role, content: m.content })),
          model: selectedModel,
          thinkingLevel,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Failed to generate response.');
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.text,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        modelUsed: data.modelUsed,
      };

      const finalMessages = [...updatedMessages, assistantMessage];
      setConversations(conversations.map(c =>
        c.id === currentConversationId
          ? { ...c, messages: finalMessages }
          : c
      ));
    } catch (err: any) {
      setError(err.message || 'Communication error with Gemini API.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#07080a] text-slate-100 overflow-hidden font-['Plus_Jakarta_Sans',sans-serif]">
      {/* Sidebar Navigation */}
      <Sidebar
        currentTab={currentTab}
        setTab={setTab}
        conversations={conversations}
        currentConversationId={currentConversationId}
        selectConversation={selectConversation}
        newChat={newChat}
        deleteConversation={deleteConversation}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        <TopBar
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          thinkingLevel={thinkingLevel}
          setThinkingLevel={setThinkingLevel}
        />

        {currentTab === 'chat' && (
          <ChatWorkspace
            messages={currentConversation?.messages || []}
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            error={error}
            selectedModel={selectedModel}
            thinkingLevel={thinkingLevel}
          />
        )}

        {currentTab === 'create' && <CreateStudioWorkspace />}
        {currentTab === 'learning' && <LearningStudioWorkspace />}
        {currentTab === 'code' && <CodeIntelligenceWorkspace />}
        {currentTab === 'knowledge' && <KnowledgeLibraryWorkspace />}
        {currentTab === 'ml' && <MlArchitectureWorkspace />}
        {currentTab === 'settings' && <SettingsModal />}
      </div>
    </div>
  );
}
