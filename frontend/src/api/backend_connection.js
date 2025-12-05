const BASE_URL = 'http://127.0.0.1:8000'

export async function sendQuery(query) {
  const response = await fetch(`${BASE_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text: query }),
  })

  const data = await response.json()

  return data
}
