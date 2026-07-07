require("dotenv").config();
const express = require("express");
const cors = require("cors");
const fetch = require("node-fetch");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: "15mb" })); // higher limit so base64 images fit
app.use(express.static(path.join(__dirname, "public")));

// ==================== CONFIGURATION ====================
const PROVIDERS = {
  gemini: {
    name: "Gemini",
    endpoint: "https://generativelanguage.googleapis.com/v1beta/models",
    models: ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
    free: true
  },
  groq: {
    name: "Groq",
    endpoint: "https://api.groq.com/openai/v1/chat/completions",
    models: ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
    free: true
  },
  openrouter: {
    name: "OpenRouter",
    endpoint: "https://openrouter.ai/api/v1/chat/completions",
    models: [
      "openrouter/free",
      "openai/gpt-oss-120b:free",
      "openai/gpt-oss-20b:free",
      "nvidia/nemotron-3-super-120b-a12b:free",
      "nvidia/nemotron-3-nano-30b-a3b:free",
      "z-ai/glm-4.5-air:free",
      "google/gemma-4-31b-it:free",
      "moonshotai/kimi-k2.6:free"
    ],
    free: true,
    docs: "https://openrouter.ai/docs"
  },
  grok: {
    name: "Grok",
    endpoint: "https://api.x.ai/v1/chat/completions",
    models: ["grok-4.3"],
    free: false,
    docs: "https://docs.x.ai"
  }
};

// ==================== STATUS ENDPOINT ====================
app.get("/api/status", (req, res) => {
  res.json({
    geminiConfigured: Boolean(process.env.GEMINI_API_KEY),
    groqConfigured: Boolean(process.env.GROQ_API_KEY),
    openrouterConfigured: Boolean(process.env.OPENROUTER_API_KEY),
    grokConfigured: Boolean(process.env.GROK_API_KEY),
    providers: PROVIDERS
  });
});

// ==================== MODELS ENDPOINT ====================
app.get("/api/models", (req, res) => {
  const availableModels = {};
  for (const [key, provider] of Object.entries(PROVIDERS)) {
    const apiKey = process.env[`${key.toUpperCase()}_API_KEY`];
    if (apiKey) {
      availableModels[key] = provider.models;
    }
  }
  res.json(availableModels);
});

// ==================== MAIN CHAT ENDPOINT ====================
app.post("/api/chat", async (req, res) => {
  const { provider, model, messages } = req.body;

  // Validation
  if (!provider || !model || !Array.isArray(messages)) {
    return res.status(400).json({ error: "Missing provider, model, or messages." });
  }

  // Check if provider exists
  if (!PROVIDERS[provider]) {
    return res.status(400).json({ error: `Unknown provider: ${provider}` });
  }

  // Check if API key is configured
  const apiKey = process.env[`${provider.toUpperCase()}_API_KEY`];
  if (!apiKey) {
    return res.status(401).json({ error: `${PROVIDERS[provider].name} API key is not configured.` });
  }

  try {
    let reply;
    if (provider === "gemini") {
      reply = await callGemini(model, messages, apiKey);
    } else if (provider === "groq" || provider === "openrouter" || provider === "grok") {
      // Both use OpenAI-compatible format
      reply = await callOpenAICompatible(model, messages, PROVIDERS[provider].endpoint, apiKey, provider);
    } else {
      return res.status(400).json({ error: `Unknown provider: ${provider}` });
    }
    
    res.json({ reply });
  } catch (err) {
    console.error(`[${provider}] Error:`, err.message);
    res.status(500).json({ 
      error: err.message || "Something went wrong calling the AI provider.",
      provider: provider
    });
  }
});

// ==================== STREAMING ENDPOINT ====================
app.post("/api/chat/stream", async (req, res) => {
  const { provider, model, messages } = req.body;

  // Validation
  if (!provider || !model || !Array.isArray(messages)) {
    return res.status(400).json({ error: "Missing provider, model, or messages." });
  }

  if (!PROVIDERS[provider]) {
    return res.status(400).json({ error: `Unknown provider: ${provider}` });
  }

  const apiKey = process.env[`${provider.toUpperCase()}_API_KEY`];
  if (!apiKey) {
    return res.status(401).json({ error: `${PROVIDERS[provider].name} API key is not configured.` });
  }

  try {
    // Only Groq and OpenRouter support streaming via OpenAI-compatible endpoint
    if (provider === "gemini") {
      // Gemini doesn't support streaming in this implementation
      // Fallback to non-streaming
      const reply = await callGemini(model, messages, apiKey);
      return res.json({ reply });
    }

    // Set up SSE (Server-Sent Events)
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no'); // Disable nginx buffering

    // Prepare the request
    const endpoint = PROVIDERS[provider].endpoint;
    const formattedMessages = formatMessagesForOpenAI(messages);
    
    const requestBody = {
      model,
      messages: formattedMessages,
      stream: true,
    };

    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMsg = errorData?.error?.message || errorData?.error || `API error (${response.status})`;
      throw new Error(errorMsg);
    }

    // Stream the response
    let buffer = '';
    let fullResponse = '';

    response.body.on('data', (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            res.write('data: [DONE]\n\n');
            return;
          }
          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices?.[0]?.delta?.content;
            
            if (content) {
              fullResponse += content;
              res.write(`data: ${JSON.stringify({ content })}\n\n`);
            }
          } catch (e) {
            // Skip invalid JSON
          }
        }
      }
    });

    response.body.on('end', () => {
      res.end();
    });

    response.body.on('error', (err) => {
      console.error('Stream error:', err);
      res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
      res.end();
    });

  } catch (err) {
    console.error(`[${provider}] Stream error:`, err.message);
    res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
    res.end();
  }
});

// ==================== HELPERS ====================

/**
 * Call Gemini API
 */
async function callGemini(model, messages, apiKey) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
  
  const contents = messages.map((m) => {
    const parts = [];
    if (m.content) parts.push({ text: m.content });
    (m.images || []).forEach((img) => {
      parts.push({ 
        inline_data: { 
          mime_type: img.mimeType || "image/jpeg", 
          data: img.base64 
        } 
      });
    });
    return { role: m.role === "user" ? "user" : "model", parts };
  });

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contents }),
  });

  const data = await response.json();
  
  if (!response.ok) {
    throw new Error(data?.error?.message || `Gemini API error (${response.status})`);
  }
  
  const text = data?.candidates?.[0]?.content?.parts?.map((p) => p.text).join("");
  if (!text) throw new Error("Gemini returned no text (possibly blocked by safety filters).");
  
  return text;
}

/**
 * Format messages for OpenAI-compatible APIs (Groq, OpenRouter)
 */
function formatMessagesForOpenAI(messages) {
  return messages.map((m) => {
    // Handle messages with images
    if (m.images && m.images.length) {
      const content = [];
      if (m.content) content.push({ type: "text", text: m.content });
      m.images.forEach((img) => {
        content.push({ 
          type: "image_url", 
          image_url: { 
            url: `data:${img.mimeType || "image/jpeg"};base64,${img.base64}` 
          } 
        });
      });
      return { 
        role: m.role === "user" ? "user" : "assistant", 
        content 
      };
    }
    
    // Handle text-only messages
    return { 
      role: m.role === "user" ? "user" : "assistant", 
      content: m.content || "" 
    };
  });
}

/**
 * Call OpenAI-compatible APIs (Groq, OpenRouter)
 */
async function callOpenAICompatible(model, messages, endpoint, apiKey, provider) {
  const formattedMessages = formatMessagesForOpenAI(messages);
  
  const requestBody = {
    model,
    messages: formattedMessages,
  };

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(requestBody),
  });

  const data = await response.json();
  
  if (!response.ok) {
    // Enhanced error handling
    const errorMsg = data?.error?.message || data?.error || `API error (${response.status})`;
    
    if (provider === "openrouter") {
      if (response.status === 401) throw new Error("❌ Invalid OpenRouter API key");
      if (response.status === 429) throw new Error("⏳ OpenRouter rate limit exceeded. Please wait and try again.");
      if (response.status === 402) throw new Error("💳 OpenRouter requires credits for this model. Try a :free model.");
    }
    
    if (provider === "groq") {
      if (response.status === 401) throw new Error("❌ Invalid Groq API key");
      if (response.status === 429) throw new Error("⏳ Groq rate limit exceeded. Please wait and try again.");
    }
    
    throw new Error(errorMsg);
  }
  
  const text = data?.choices?.[0]?.message?.content;
  if (!text) throw new Error("No reply returned from the API.");
  
  // Log token usage for monitoring
  if (data?.usage) {
    console.log(`[${provider}] Usage - Prompt: ${data.usage.prompt_tokens}, Completion: ${data.usage.completion_tokens}, Total: ${data.usage.total_tokens}`);
  }
  
  return text;
}

// ==================== FALLBACK ROUTE ====================
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// ==================== START SERVER ====================
app.listen(PORT, () => {
  console.log(`🤖 AI Multi-Console running at http://localhost:${PORT}`);
  console.log(`📊 Status:`);
  console.log(`   Gemini: ${Boolean(process.env.GEMINI_API_KEY) ? "✅" : "❌"}`);
  console.log(`   Groq: ${Boolean(process.env.GROQ_API_KEY) ? "✅" : "❌"}`);
  console.log(`   OpenRouter: ${Boolean(process.env.OPENROUTER_API_KEY) ? "✅" : "❌"}`);
  console.log(`\n🔥 OpenRouter Free Models:`);
  console.log(`   • meta-llama/llama-3-70b-instruct:free (Best)`);
  console.log(`   • deepseek/deepseek-chat:free (DeepSeek FREE!)`);
  console.log(`   • mistralai/mistral-7b-instruct:free (Fast)`);
  console.log(`   • microsoft/phi-3-mini-128k-instruct:free (128K context)`);
});