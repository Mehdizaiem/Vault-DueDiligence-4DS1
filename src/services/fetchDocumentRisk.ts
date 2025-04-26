// src/services/fetchDocumentRisk.ts
export async function fetchDocumentRisk(documentId: string) {
    const query = `
      {
        Get {
          User_doc_risk(
            where: {
              path: ["document_id"],
              operator: Equal,
              valueText: "${documentId}"
            }
          ) {
            document_id
            title
            risk_score
            risk_category
            risk_factors
          }
        }
      }
    `;
  
    const res = await fetch("http://localhost:9090/v1/graphql", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
  
    const json = await res.json();
    const results = json.data?.Get?.User_doc_risk || [];
  
    if (results.length === 0) {
      throw new Error("No risk data found for document.");
    }
  
    return results[0]; // Return the first (and should be only) match
  }
  