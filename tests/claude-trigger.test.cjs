const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const workflow = fs.readFileSync(process.env.CLAUDE_WORKFLOW || path.join(__dirname, '../.github/workflows/claude.yml'), 'utf8').replace(/\r\n/g, '\n');
const expression = workflow.match(/    if: \|\n([\s\S]*?)(?=    runs-on:)/)[1];
const gate = new vm.Script(`(${expression})`);
function permitted(event_name, commentAuthor, issueAuthor, mention = '@claude') {
  return gate.runInNewContext({
    github: { event_name, event: {
      comment: { body: mention, author_association: commentAuthor },
      issue: { body: mention, author_association: issueAuthor },
    } },
    contains: (value, needle) => value?.includes(needle) || false,
    fromJSON: JSON.parse,
  });
}

for (const event of ['issue_comment', 'pull_request_review_comment']) {
  test(`${event}: outsider cannot inherit an issue owner's trust`, () => {
    for (const trustedIssueAuthor of ['OWNER', 'MEMBER', 'COLLABORATOR']) {
      assert.equal(permitted(event, 'NONE', trustedIssueAuthor), false);
      assert.equal(permitted(event, 'CONTRIBUTOR', trustedIssueAuthor), false);
    }
  });
  test(`${event}: trusted commenters can explicitly invoke Claude`, () => {
    for (const author of ['OWNER', 'MEMBER', 'COLLABORATOR']) {
      assert.equal(permitted(event, author, 'NONE'), true);
      assert.equal(permitted(event, author, 'OWNER', 'ordinary comment'), false);
    }
  });
}
test('issue events authorize the issue author and require an explicit mention', () => {
  for (const author of ['OWNER', 'MEMBER', 'COLLABORATOR']) {
    assert.equal(permitted('issues', undefined, author), true);
    assert.equal(permitted('issues', undefined, author, 'ordinary issue'), false);
  }
  assert.equal(permitted('issues', undefined, 'NONE'), false);
});
test('unrelated events never invoke Claude', () => {
  assert.equal(permitted('push', 'OWNER', 'OWNER'), false);
});
