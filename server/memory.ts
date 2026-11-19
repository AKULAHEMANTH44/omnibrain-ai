import { DatabaseSync } from 'node:sqlite';
import fs from 'fs';
import path from 'path';

export interface StoredMessage {
  id: string;
  conversationId: string;
  role: string;
  content: string;
  timestamp: string;
}

export interface StoredConversation {
  id: string;
  title: string;
  updatedAt: string;
}

export class MemoryEngine {
  private db: DatabaseSync | null = null;
  private inMemoryFallback: Map<string, StoredMessage[]> = new Map();

  constructor(dbPath = './omnibrain.db') {
    try {
      const dir = path.dirname(dbPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      this.db = new DatabaseSync(dbPath);
      this.initTables();
    } catch (err) {
      console.warn('SQLite DatabaseSync not available or failed to init. Using in-memory fallback.', err);
      this.db = null;
    }
  }

  private initTables() {
    if (!this.db) return;
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        updatedAt TEXT
      );
      CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversationId TEXT,
        role TEXT,
        content TEXT,
        timestamp TEXT,
        FOREIGN KEY(conversationId) REFERENCES conversations(id)
      );
    `);
  }

  saveConversation(id: string, title: string) {
    const updatedAt = new Date().toISOString();
    if (this.db) {
      const stmt = this.db.prepare(`
        INSERT INTO conversations (id, title, updatedAt)
        VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET title = ?, updatedAt = ?
      `);
      stmt.run(id, title, updatedAt, title, updatedAt);
    }
  }

  saveMessage(msg: StoredMessage) {
    if (this.db) {
      const stmt = this.db.prepare(`
        INSERT OR REPLACE INTO messages (id, conversationId, role, content, timestamp)
        VALUES (?, ?, ?, ?, ?)
      `);
      stmt.run(msg.id, msg.conversationId, msg.role, msg.content, msg.timestamp);
    } else {
      const list = this.inMemoryFallback.get(msg.conversationId) || [];
      list.push(msg);
      this.inMemoryFallback.set(msg.conversationId, list);
    }
  }

  getMessages(conversationId: string): StoredMessage[] {
    if (this.db) {
      const stmt = this.db.prepare(`SELECT * FROM messages WHERE conversationId = ? ORDER BY timestamp ASC`);
      return stmt.all(conversationId) as unknown as StoredMessage[];
    }
    return this.inMemoryFallback.get(conversationId) || [];
  }

  clear() {
    if (this.db) {
      this.db.exec(`DELETE FROM messages; DELETE FROM conversations;`);
    }
    this.inMemoryFallback.clear();
  }
}

export const globalMemoryEngine = new MemoryEngine();
