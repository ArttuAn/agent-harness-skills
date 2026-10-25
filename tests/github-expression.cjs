// A deliberately small, non-executing parser for the workflow's GitHub condition.
// Unsupported syntax fails the test; workflow text is never JavaScript code.
function evaluateExpression(source, values) {
  const tokens = [];
  let cursor = 0;
  while (cursor < source.length) {
    if (/\s/.test(source[cursor])) { cursor++; continue; }
    const operator = source.slice(cursor, cursor + 2);
    if (['&&', '||', '=='].includes(operator)) {
      tokens.push({type: operator}); cursor += 2; continue;
    }
    const char = source[cursor];
    if ('(),'.includes(char)) { tokens.push({type: char}); cursor++; continue; }
    if (char === "'" || char === '"') {
      const quote = char;
      let value = '', closed = false;
      cursor++;
      while (cursor < source.length) {
        const next = source[cursor++];
        if (next === quote) { closed = true; break; }
        if (next === '\\') {
          const escaped = source[cursor++];
          if (escaped !== quote && escaped !== '\\') throw Error('Unsupported string escape');
          value += escaped;
        } else value += next;
      }
      if (!closed) throw Error('Unclosed string');
      tokens.push({type: 'string', value});
      continue;
    }
    const identifier = source.slice(cursor).match(/^[A-Za-z_][A-Za-z0-9_.]*/)?.[0];
    if (!identifier) throw Error('Unsupported expression syntax');
    tokens.push({type: 'identifier', value: identifier});
    cursor += identifier.length;
  }
  tokens.push({type: 'end'});
  cursor = 0;
  const take = type => tokens[cursor].type === type && tokens[cursor++];
  const expect = type => { if (!take(type)) throw Error(`Expected ${type}`); };

  function value() {
    if (take('(')) { const result = or(); expect(')'); return result; }
    const literal = take('string');
    if (literal) return literal.value;
    const name = take('identifier')?.value;
    if (!name) throw Error('Expected expression value');
    if (take('(')) {
      const args = [];
      if (tokens[cursor].type !== ')') {
        args.push(or());
        while (take(',')) args.push(or());
      }
      expect(')');
      if (name === 'fromJSON' && args.length === 1 && typeof args[0] === 'string') return JSON.parse(args[0]);
      if (name === 'contains' && args.length === 2) {
        const [haystack, needle] = args;
        return (typeof haystack === 'string' || Array.isArray(haystack)) && haystack.includes(needle);
      }
      throw Error('Unsupported expression function');
    }
    if (!Object.hasOwn(values, name)) throw Error('Unsupported expression identifier');
    return values[name];
  }
  function equal() {
    let result = value();
    while (take('==')) result = result === value();
    return result;
  }
  function and() {
    let result = equal();
    while (take('&&')) { const next = equal(); result = Boolean(result && next); }
    return result;
  }
  function or() {
    let result = and();
    while (take('||')) { const next = and(); result = Boolean(result || next); }
    return result;
  }
  const result = or();
  expect('end');
  return Boolean(result);
}

module.exports = {evaluateExpression};
