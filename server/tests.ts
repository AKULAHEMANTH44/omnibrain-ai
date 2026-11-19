import { classifyQuestionCategory, SPECIALIZED_MODULES } from './modules/registry';
import { globalKnowledgeEngine } from './knowledge';
import { globalMemoryEngine } from './memory';
import { globalOrchestrator } from './orchestrator';
import { OmnibrainNativeProvider } from './providers/omnibrain';

export async function runOmnibrainTests() {
  console.log('🧪 Starting OMNIBRAIN AI Architecture Test Suite...\n');
  const results: Array<{ test: string; passed: boolean; details: string }> = [];

  // 1. Question Classification Test
  try {
    const catMath = classifyQuestionCategory('Prove the Riemann hypothesis and calculate integrals.');
    const catCode = classifyQuestionCategory('Write a Python function to sort a binary tree.');
    const passed = catMath === 'mathematics' && catCode === 'programming';
    results.push({ test: 'Question Classification & Routing', passed, details: `Math -> ${catMath}, Code -> ${catCode}` });
  } catch (err: any) {
    results.push({ test: 'Question Classification & Routing', passed: false, details: err.message });
  }

  // 2. Specialized Modules Test
  try {
    const count = Object.keys(SPECIALIZED_MODULES).length;
    const passed = count === 10 && SPECIALIZED_MODULES.mathematics !== undefined;
    results.push({ test: 'Specialized Modules (10 Modules)', passed, details: `Total modules registered: ${count}` });
  } catch (err: any) {
    results.push({ test: 'Specialized Modules (10 Modules)', passed: false, details: err.message });
  }

  // 3. Knowledge Engine Retrieval Test
  try {
    const resultsSearch = globalKnowledgeEngine.search('Transformer scaling laws', 1);
    const passed = resultsSearch.length > 0 && resultsSearch[0].documentTitle.includes('Transformer');
    results.push({ test: 'Knowledge Engine Vector Retrieval', passed, details: `Retrieved: ${resultsSearch[0]?.documentTitle || 'None'}` });
  } catch (err: any) {
    results.push({ test: 'Knowledge Engine Vector Retrieval', passed: false, details: err.message });
  }

  // 4. Memory Persistence Test
  try {
    const testConvId = 'test_conv_999';
    globalMemoryEngine.saveConversation(testConvId, 'Test Conversation');
    globalMemoryEngine.saveMessage({
      id: 'msg_1',
      conversationId: testConvId,
      role: 'user',
      content: 'Hello Omnibrain',
      timestamp: new Date().toISOString(),
    });
    const msgs = globalMemoryEngine.getMessages(testConvId);
    const passed = msgs.length === 1 && msgs[0].content === 'Hello Omnibrain';
    results.push({ test: 'SQLite Memory Persistence', passed, details: `Stored and retrieved ${msgs.length} message(s)` });
  } catch (err: any) {
    results.push({ test: 'SQLite Memory Persistence', passed: false, details: err.message });
  }

  // 5. Independent Native Model Provider Test (OMNIBRAIN CORE)
  try {
    const nativeProvider = new OmnibrainNativeProvider();
    const isAvail = await nativeProvider.isAvailable();
    const genRes = await nativeProvider.generate({
      model: 'omnibrain-transformer-xl-v1',
      messages: [{ role: 'user', content: 'Test prompt for native model' }],
    });
    const passed = isAvail && genRes.text.length > 0 && genRes.provider === 'OmnibrainNative';
    results.push({ test: 'Independent OMNIBRAIN Native Core', passed, details: `Generated ${genRes.text.length} chars via CPU inference` });
  } catch (err: any) {
    results.push({ test: 'Independent OMNIBRAIN Native Core', passed: false, details: err.message });
  }

  // Summary Report
  console.log('📋 Test Results Summary:');
  let allPassed = true;
  results.forEach((r, idx) => {
    const icon = r.passed ? '✅' : '❌';
    console.log(`${idx + 1}. ${icon} [${r.test}]: ${r.details}`);
    if (!r.passed) allPassed = false;
  });

  console.log(`\nResult: ${allPassed ? 'ALL TESTS PASSED SUCCESSFULLY 🎉' : 'SOME TESTS FAILED ⚠️'}\n`);
  return { allPassed, results };
}
