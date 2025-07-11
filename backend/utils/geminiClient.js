
const { GoogleGenerativeAI } = require('@google/generative-ai');
const config = require('../config');

// Initialize the Google Generative AI with API key
const genAI = new GoogleGenerativeAI(config.gemini.apiKey);

// Store chat history for different users to maintain context
const chatHistory = new Map();

// Maximum number of messages to keep in history
const MAX_HISTORY_LENGTH = 20;

async function generateChatResponse(userId, userMessage, userRole = 'user', modelName = 'gemini-1.5-flash') {
  try {
    // Get or initialize chat history for this user
    if (!chatHistory.has(userId)) {
      chatHistory.set(userId, []);
    }
    
    const userHistory = chatHistory.get(userId);
    
    // Create the chat model
    const model = genAI.getGenerativeModel({ model: modelName });
    const chat = model.startChat({
      history: userHistory,
      generationConfig: {
        maxOutputTokens: 2000,
        temperature: 0.7,
        topP: 0.95,
        topK: 64,
      },
    });

    // Set system instructions based on user role
    let systemInstruction;
    if (userRole === 'admin' || userRole === 'super_admin') {
      systemInstruction = `You are SafeStreets Professional AI Assistant, an expert consultant designed to assist administrators and municipal managers with road infrastructure management.

Professional Guidelines:
1. Provide concise, actionable insights focused on administrative decision-making.
2. Use professional terminology relevant to urban planning, civil engineering, and public works administration.
3. Present information in structured formats with clear prioritization when applicable.
4. Offer data-driven recommendations for resource allocation, maintenance scheduling, and budgeting.
5. When appropriate, reference industry best practices and standards for road maintenance.
6. Balance technical detail with executive-level summary information.
7. Always maintain a professional, authoritative tone while remaining helpful and clear.

Primary areas of expertise:
- Road damage assessment metrics and classification systems
- Maintenance prioritization frameworks and decision matrices
- Infrastructure lifecycle management and preventive maintenance scheduling
- Municipal resource optimization and workflow efficiency
- Data-driven infrastructure planning and budgeting
- Regulatory compliance for public infrastructure

When providing recommendations, consider:
- Budget constraints and resource allocation efficiency
- Public safety implications and risk mitigation
- Long-term infrastructure sustainability
- Weather and environmental impact factors
- Traffic patterns and usage intensity`;
    } else if (userRole === 'field_worker') {
      systemInstruction = `You are SafeStreets Field Operations AI Assistant, designed to support workers during on-site road damage assessment and repair operations.

Field Support Guidelines:
1. Provide clear, practical guidance with minimal technical jargon.
2. Focus on actionable advice for identifying damage types, assessing severity, and recording accurate reports.
3. Offer step-by-step procedures when relevant.
4. Prioritize safety considerations and proper assessment techniques.
5. Use direct, concise language focused on immediate practical needs.
6. When appropriate, suggest efficient field workflows and documentation best practices.

Primary areas of support:
- Visual identification of damage types (potholes, cracks, erosion, etc.)
- Severity assessment guidelines and measurement techniques
- Safety protocols for roadside work and assessment
- Efficient documentation and reporting procedures
- Equipment usage recommendations
- Weather considerations for field operations
- Prioritization of multiple damage instances

For field questions, emphasize:
- Practical, implementable solutions
- Safety-first approach
- Consistency in assessment methodology
- Efficient documentation practices`;
    } else {
      systemInstruction = `You are SafeStreets AI Assistant, designed to provide helpful information about road maintenance and the SafeStreets system.

Guidelines:
1. Provide clear, accurate information about road conditions and maintenance.
2. Explain concepts in easy-to-understand language without technical jargon.
3. Focus on practical information that's relevant to everyday road users.
4. Be concise while providing complete answers to questions.
5. When appropriate, offer safety tips and best practices.

Primary areas of assistance:
- Explaining the SafeStreets reporting system
- Describing different types of road damage
- Providing general information about road maintenance processes
- Safety tips for navigating damaged roads
- How to effectively report road issues`;
    }

    // Add system message if history is empty
    if (userHistory.length === 0) {
      userHistory.push({
        role: 'model',
        parts: [{ text: 'I am SafeStreets AI Assistant. How can I help you today?' }],
      });
    }

    // Send the message
    const result = await chat.sendMessage([
      { text: `${systemInstruction}\n\nUser message: ${userMessage}` }
    ]);
    
    const response = result.response;
    const responseText = response.text();
    
    // Update history with this interaction
    userHistory.push({
      role: 'user',
      parts: [{ text: userMessage }],
    });
    
    userHistory.push({
      role: 'model',
      parts: [{ text: responseText }],
    });
    
    // Trim history if it gets too long
    if (userHistory.length > MAX_HISTORY_LENGTH) {
      // Remove oldest interactions (keep system message)
      userHistory.splice(1, 2); // Remove oldest user+model pair
    }
    
    return responseText;
  } catch (error) {
    console.error('Error generating chat response:', error);
    throw new Error(`Failed to generate chat response: ${error.message}`);
  }
}


async function analyzeRoadDamageImage(imageBase64, damageInfo) {
  try {
    const model = genAI.getGenerativeModel({ 
      model: 'gemini-1.5-pro', 
      generationConfig: {
        temperature: 0.2,
        topP: 0.95,
        topK: 64,
      }
    });
    
    const prompt = `
    You are SafeStreets Professional Assessment System, an expert in civil engineering and road infrastructure analysis.
    
    Analyze this road damage image with professional civil engineering expertise and provide a detailed technical assessment for municipal administrators.
    
    Given information:
    - Location: ${damageInfo.location || 'Unknown'}
    - Reported damage type: ${damageInfo.damageType || 'Unknown'}
    - Reported severity: ${damageInfo.severity || 'Unknown'}
    
    Please provide a structured JSON response with the following fields:
    1. "damageType": Specific classification of damage using professional terminology (e.g., "Class 2 Longitudinal Cracking", "Deep Pothole with Edge Deterioration")
    2. "severity": Precise severity assessment (Low, Medium-Low, Medium, Medium-High, High) with numeric rating if possible (e.g., "High - 8/10")
    3. "maintenanceCategory": Infrastructure maintenance classification (e.g., "Preventive", "Routine", "Structural", "Emergency")
    4. "recommendedAction": Specific professional maintenance protocol or technique recommended
    5. "materialRequirements": Estimated materials needed for repair
    6. "estimatedRepairTime": Professional time estimate including crew size assumption (e.g., "4-6 hours with 3-person crew")
    7. "estimatedCost": Cost range estimate based on standard municipal rates (if reliable indicators present)
    8. "trafficImpact": Assessment of traffic disruption during repair (Minimal, Moderate, Significant)
    9. "safetyRisk": Safety risk assessment if left unaddressed (Low, Medium, High)
    10. "analysis": Concise professional assessment using civil engineering terminology (max 150 words)
    11. "preventativeMeasures": Recommendations to prevent recurrence
    
    Ensure your analysis reflects professional civil engineering standards and municipal infrastructure management best practices. Use appropriate technical terminology while maintaining clarity for administrative decision-makers.
    `;

    const result = await model.generateContent([
      prompt,
      { inlineData: { mimeType: 'image/jpeg', data: imageBase64 } }
    ]);

    const response = result.response;
    const responseText = response.text();
    
    // Try to parse JSON response
    try {
      return JSON.parse(responseText);
    } catch (e) {
      console.error('Failed to parse AI response as JSON:', e);
      // Return text response as fallback
      return {
        analysis: responseText,
        damageType: damageInfo.damageType || 'Unknown',
        severity: damageInfo.severity || 'Unknown',
        recommendedAction: 'Please review the analysis text',
        estimatedRepairTime: 'Unknown'
      };
    }
  } catch (error) {
    console.error('Error analyzing road damage image:', error);
    throw new Error(`Failed to analyze image: ${error.message}`);
  }
}

function clearChatHistory(userId) {
  chatHistory.delete(userId);
}

module.exports = {
  generateChatResponse,
  analyzeRoadDamageImage,
  clearChatHistory
};
