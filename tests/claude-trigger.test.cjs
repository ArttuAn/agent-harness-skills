const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const { evaluateExpression } = require('./github-expression.cjs');

// Test repository-owned code only; an environment-selected file must not
// become input to the workflow condition parser.
const workflow = fs.readFileSync(path.join(__dirname, '../.github/workflows/claude.yml'), 'utf8').replace(/\r\n/g, '\n');
const expression = workflow.match(/    if: \|\n([\s\S]*?)(?=    runs-on:)/)[1];
function permitted(event_name, commentAuthor, issueAuthor, mention = '@claude') {
  return evaluateExpression(expression, {
    'github.event_name': event_name,
    'github.event.comment.body': mention,
    'github.event.comment.author_association': commentAuthor,
    'github.event.issue.body': mention,
    'github.event.issue.author_association': issueAuthor,
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

test('unsupported functions and executable syntax are rejected', () => {
  for (const source of ['process.exit()', 'require("fs")', 'contains.constructor("return 1")()', 'x; y']) {
    assert.throws(() => evaluateExpression(source, {}), /Unsupported/);
  }
});

test('event bodies are data and cannot introduce expression syntax', () => {
  assert.equal(permitted('issue_comment', 'NONE', 'OWNER', "@claude'); process.exit()"), false);
  assert.equal(permitted('issue_comment', 'OWNER', 'NONE', "@claude'); process.exit()"), true);
});
