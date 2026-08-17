import { supabase } from '@/api/supabaseClient';

/**
 * Drop-in замена base44.integrations.Core.InvokeLLM({ prompt, response_json_schema }).
 * Вызывает Edge Function 'invoke-llm', которая держит ключ GigaChat на сервере.
 *
 * Раньше:  base44.integrations.Core.InvokeLLM({ prompt, response_json_schema })
 * Теперь:  invokeLLM({ prompt, response_json_schema })
 */
export async function invokeLLM({ prompt, response_json_schema }) {
  const { data, error } = await supabase.functions.invoke('invoke-llm', {
    body: { prompt, response_json_schema },
  });
  if (error) throw error;
  if (data?.error) throw new Error(data.error);
  return data;
}
